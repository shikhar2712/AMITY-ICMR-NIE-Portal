"""
Dashboard Module
================
Renders the two admin screens added for the "Dashboard" task:

  1. "Dashboard"     -> Dashboard - Patient Information  (six KPI cards)
  2. "View Records"  -> View - Patient Information        (records table with
                        New case / View / Edit / Delete(soft) / Update DR)

This module only READS and EDITS saved prediction records through the
MongoDB-backed DataHandler. It never touches the model or the .pth files.
"""
from datetime import datetime

import streamlit as st

from data_handler import (
    get_dashboard_metrics,
    get_records,
    get_record,
    update_patient_record_in_db,
    soft_delete_record_in_db,
    save_doctor_lab_data_to_db,
)

# Virus options for the Doctor-Recommendation / Lab multiselects. model_handler
# is already imported (and its mappings populated) by app.py at runtime; the
# dicts are mutated in place by refresh_virus_mappings(), so this stays current.
try:
    from model_handler import VIRUS_MAPPING, OTHER_VIRUS_MAPPING
except Exception:  # defensive: tooling without torch installed
    VIRUS_MAPPING, OTHER_VIRUS_MAPPING = {}, {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _virus_options():
    return sorted(set(list(VIRUS_MAPPING.values()) + list(OTHER_VIRUS_MAPPING.values())))


def _fmt(value):
    """Human-friendly cell value for the read-only tables."""
    if value in (None, "", []):
        return "—"
    if isinstance(value, list):
        return ", ".join(str(x) for x in value) if value else "—"
    return str(value)


def _set_nav(page_name):
    """Callback: switch the sidebar radio to another page (safe inside callbacks)."""
    st.session_state['navigation_page'] = page_name


def _open_action(action, record_id):
    st.session_state['vr_action'] = action
    st.session_state['vr_record_id'] = record_id


def _close_action():
    st.session_state.pop('vr_action', None)
    st.session_state.pop('vr_record_id', None)


def _flash(message):
    st.session_state['vr_flash'] = message


# ===========================================================================
# 1) DASHBOARD PAGE - KPI cards
# ===========================================================================
_CARD_CSS = """
<style>
.kpi-row { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:16px; }
.kpi-card { flex:1; min-width:190px; border-radius:8px; padding:18px 20px; color:#fff;
            box-shadow:0 2px 4px rgba(0,0,0,.18); }
.kpi-card .kpi-value { font-size:2.5rem; font-weight:700; line-height:1.05; }
.kpi-card .kpi-label { font-size:.95rem; opacity:.95; margin-top:6px; }
.kpi-blue{background:#3c8dbc;} .kpi-green{background:#00a65a;} .kpi-red{background:#dd4b39;}
.kpi-aqua{background:#00c0ef;} .kpi-yellow{background:#f39c12;} .kpi-purple{background:#605ca8;}
</style>
"""


def _kpi_card(value, label, css_class):
    return (f'<div class="kpi-card {css_class}">'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div></div>')


def render_dashboard_page():
    st.title("📊 Dashboard – Patient Information")
    st.caption("Live summary of enrolled cases and doctor-recommendation status.")

    metrics = get_dashboard_metrics()

    st.markdown(_CARD_CSS, unsafe_allow_html=True)
    # Row 1 - enrolment / doctor-recommendation status
    st.markdown(
        '<div class="kpi-row">'
        + _kpi_card(metrics.get('enrolled', 0), "No. of Records Enrolled", "kpi-blue")
        + _kpi_card(metrics.get('dr_completed', 0), "Doctor Recommendations Completed", "kpi-green")
        + _kpi_card(metrics.get('dr_pending', 0), "Pending for Doctor Recommendation", "kpi-red")
        + '</div>',
        unsafe_allow_html=True,
    )
    # Row 2 - enrolment volume
    st.markdown(
        '<div class="kpi-row">'
        + _kpi_card(metrics.get('daily', 0), "Daily Enrolled (today)", "kpi-aqua")
        + _kpi_card(metrics.get('weekly', 0), "Weekly Enrolled (last 7 days)", "kpi-yellow")
        + _kpi_card(metrics.get('monthly', 0), "Monthly Enrolled (last 30 days)", "kpi-purple")
        + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    c1, c2, _ = st.columns([1, 1, 3])
    c1.button("➕ New Case", type="primary", use_container_width=True,
              on_click=_set_nav, args=("Prediction",), key="dash_new_case")
    c2.button("🗂️ View Records", use_container_width=True,
              on_click=_set_nav, args=("View Records",), key="dash_view_records")


# ===========================================================================
# 2) VIEW RECORDS PAGE - table + per-row actions
# ===========================================================================
def render_view_records_page():
    st.title("🗂️ View – Patient Information")

    flash = st.session_state.pop('vr_flash', None)
    if flash:
        st.success(flash)

    c1, c2 = st.columns([3, 1])
    c1.caption("All enrolled patients. Use the per-row actions to view, edit, "
               "delete (soft) or complete the doctor recommendation.")
    c2.button("➕ New Case", type="primary", use_container_width=True,
              on_click=_set_nav, args=("Prediction",), key="view_new_case")

    # If an action panel is open, render it instead of the table.
    if st.session_state.get('vr_action') and st.session_state.get('vr_record_id'):
        _render_action_panel(st.session_state['vr_action'], st.session_state['vr_record_id'])
        return

    records = get_records()
    if not records:
        st.info("No patient records yet. Click **New Case** to enrol the first patient.")
        return

    _render_table(records)


def _render_table(records):
    st.markdown(f"**{len(records)} record(s)**")
    weights = [0.5, 1.3, 1.6, 1.2, 1.2, 1.1, 0.6, 0.6, 0.9, 0.6]
    headers = ["#", "Study ID", "Patient Name", "Date of Collection",
               "Date of Admission", "Status", "Edit", "Del", "Upd DR", "View"]
    head_cols = st.columns(weights)
    for col, label in zip(head_cols, headers):
        col.markdown(f"**{label}**")
    st.markdown("<hr style='margin:2px 0;border:none;border-top:1px solid #ddd;'>",
                unsafe_allow_html=True)

    for idx, rec in enumerate(records, start=1):
        rid = rec['_id']
        cols = st.columns(weights)
        cols[0].write(str(idx))
        cols[1].write(_fmt(rec.get('patient_study_id')))
        cols[2].write(_fmt(rec.get('patient_name')))
        cols[3].write(_fmt(rec.get('date_of_collection')))
        cols[4].write(_fmt(rec.get('date_of_admission')))
        completed = bool(rec.get('doctor_lab_submitted_at'))
        cols[5].markdown("🟢 Completed" if completed else "🔴 Pending")
        cols[6].button("✏️", key=f"edit_{rid}", help="Edit patient info",
                       on_click=_open_action, args=("edit", rid))
        cols[7].button("🗑️", key=f"del_{rid}", help="Delete (soft)",
                       on_click=_open_action, args=("delete", rid))
        cols[8].button("🩺", key=f"dr_{rid}", help="Update Doctor Recommendation",
                       on_click=_open_action, args=("updatedr", rid))
        cols[9].button("👁️", key=f"view_{rid}", help="View full record",
                       on_click=_open_action, args=("view", rid))


def _render_action_panel(action, record_id):
    st.button("← Back to records", on_click=_close_action, key="vr_back")
    record = get_record(record_id)
    if not record:
        st.error("Record not found (it may have been deleted).")
        return
    if action == "view":
        _render_view_detail(record)
    elif action == "edit":
        _render_edit_form(record)
    elif action == "updatedr":
        _render_update_dr_form(record)
    elif action == "delete":
        _render_delete_confirm(record)


# ---- View (read-only) ------------------------------------------------------
def _render_view_detail(rec):
    st.subheader(f"👁️ {rec.get('patient_name') or 'Patient'} — full record")
    completed = bool(rec.get('doctor_lab_submitted_at'))
    st.markdown(f"**Doctor recommendation:** {'🟢 Completed' if completed else '🔴 Pending'}")

    st.markdown("##### Patient & administrative")
    admin = {
        "Patient ID": rec.get('patient_id'), "Study ID": rec.get('patient_study_id'),
        "MRD ID": rec.get('patient_mrd_id'), "Hospital": rec.get('hospital'),
        "Department": rec.get('department'), "Department (specify)": rec.get('department_specification'),
        "Date of Collection": rec.get('date_of_collection'), "Date of Admission": rec.get('date_of_admission'),
        "Mobile": rec.get('mobile_no'),
    }
    st.table({"Field": list(admin.keys()), "Value": [_fmt(v) for v in admin.values()]})

    st.markdown("##### Demographics & clinical")
    demo = {
        "Age": rec.get('age'), "Sex": rec.get('sex'), "Patient Type": rec.get('patient_type'),
        "Onset of Illness": rec.get('onset_of_illness'), "Duration (days)": rec.get('duration_of_illness_days'),
        "State": rec.get('state_name'), "District": rec.get('district_name'),
        "Syndrome": rec.get('syndrome_name'), "Month": rec.get('month_name'),
    }
    st.table({"Field": list(demo.keys()), "Value": [_fmt(v) for v in demo.values()]})

    symptoms = [k.replace('symptom_', '').replace('_', ' ').title()
                for k, v in rec.items() if k.startswith('symptom_') and v == 'Yes']
    st.markdown("##### Symptoms reported")
    st.write(", ".join(symptoms) if symptoms else "None recorded")

    st.markdown("##### Prediction")
    pv = rec.get('predicted_virus_name')
    if pv:
        try:
            conf = float(rec.get('prediction_confidence_percent') or 0)
        except (TypeError, ValueError):
            conf = 0.0
        st.write(f"**Predicted virus:** {pv} ({conf:.1f}%)")
    else:
        st.write("No prediction stored.")

    if completed:
        st.markdown("##### Doctor recommendation & laboratory")
        dr = {
            "Recommended": rec.get('doctor_recommended_viruses'), "Lab ID": rec.get('lab_id'),
            "Test Performed": rec.get('test_performed'), "Laboratory Results": rec.get('laboratory_results'),
            "Confirmed Pathogen": rec.get('confirmed_pathogen'), "Date of Report": rec.get('date_of_report'),
        }
        st.table({"Field": list(dr.keys()), "Value": [_fmt(v) for v in dr.values()]})


# ---- Edit patient info -----------------------------------------------------
def _render_edit_form(rec):
    st.subheader(f"✏️ Edit — {rec.get('patient_name') or 'Patient'}")
    hosp_opts, dept_opts = ["MMC", "TMC"], ["Medicine", "Pediatrics", "Other"]
    sex_opts, ptype_opts = ["Female", "Male", "Other"], ["Outpatient", "Inpatient"]

    try:
        age_val = max(0, min(120, int(float(rec.get('age') or 0))))
    except (TypeError, ValueError):
        age_val = 0

    with st.form(key=f"edit_form_{rec['_id']}"):
        c1, c2 = st.columns(2)
        with c1:
            study_id = st.text_input("Patient Study ID", value=rec.get('patient_study_id') or "")
            mrd_id = st.text_input("Patient MRD ID", value=rec.get('patient_mrd_id') or "")
            name = st.text_input("Patient Name", value=rec.get('patient_name') or "")
            hospital = st.selectbox("Hospital", hosp_opts,
                                    index=hosp_opts.index(rec['hospital']) if rec.get('hospital') in hosp_opts else 0)
            dept = st.selectbox("Department", dept_opts,
                                index=dept_opts.index(rec['department']) if rec.get('department') in dept_opts else 0)
            dept_spec = st.text_input("Specify Department (if Other)",
                                      value=rec.get('department_specification') or "")
        with c2:
            age = st.number_input("Age", min_value=0, max_value=120, value=age_val, step=1)
            sex = st.selectbox("Sex", sex_opts,
                               index=sex_opts.index(rec['sex']) if rec.get('sex') in sex_opts else 0)
            ptype = st.selectbox("Patient Type", ptype_opts,
                                 index=ptype_opts.index(rec['patient_type']) if rec.get('patient_type') in ptype_opts else 0)
            mobile = st.text_input("Mobile No", value=rec.get('mobile_no') or "")
            date_coll = st.text_input("Date of Collection (DD-MM-YYYY)", value=rec.get('date_of_collection') or "")
            date_adm = st.text_input("Date of Admission (DD-MM-YYYY)", value=rec.get('date_of_admission') or "")

        if st.form_submit_button("💾 Save changes", type="primary", use_container_width=True):
            fields = {
                'patient_study_id': study_id.strip(), 'patient_mrd_id': mrd_id.strip(),
                'patient_name': name.strip(), 'hospital': hospital, 'department': dept,
                'department_specification': dept_spec.strip() if dept == "Other" else "",
                'age': int(age), 'sex': sex, 'patient_type': ptype, 'mobile_no': mobile.strip(),
                'date_of_collection': date_coll.strip(), 'date_of_admission': date_adm.strip(),
            }
            if update_patient_record_in_db(rec['_id'], fields):
                _flash("✅ Record updated.")
                _close_action()
                st.rerun()
            else:
                st.error("❌ Could not update the record.")


# ---- Update Doctor Recommendation (reuses the existing DR/lab fields) ------
def _render_update_dr_form(rec):
    st.subheader(f"🩺 Update Doctor Recommendation — {rec.get('patient_name') or 'Patient'}")
    all_opts = _virus_options()
    lab_opts = [""] + all_opts

    def _keep(values, allowed):
        return [x for x in values if x in allowed] if isinstance(values, list) else []

    res_opts = ["", "Positive", "Negative"]
    with st.form(key=f"dr_form_{rec['_id']}"):
        doctor_recommended = st.multiselect(
            "Doctor Recommended - Suspected Pathogens (up to 5)", options=all_opts,
            default=_keep(rec.get('doctor_recommended_viruses'), all_opts))
        c1, c2 = st.columns(2)
        with c1:
            lab_id = st.text_input("Lab ID", value=rec.get('lab_id') or "")
            test_performed = st.multiselect("Test Performed", options=lab_opts,
                                            default=_keep(rec.get('test_performed'), lab_opts))
            sample_type = st.text_input("Sample Type", value=rec.get('sample_type') or "")
            date_sample = st.text_input("Date of Sample Collection (DD-MM-YYYY)",
                                        value=rec.get('date_of_sample_collection') or "")
        with c2:
            diagnostic_method = st.text_input("Diagnostic Method", value=rec.get('diagnostic_method') or "")
            lab_results = st.selectbox("Laboratory Results", options=res_opts,
                                       index=res_opts.index(rec['laboratory_results'])
                                       if rec.get('laboratory_results') in res_opts else 0)
            confirmed = st.multiselect("Confirmed Pathogen", options=lab_opts,
                                       default=_keep(rec.get('confirmed_pathogen'), lab_opts))
            date_report = st.text_input("Date of Report (DD-MM-YYYY)", value=rec.get('date_of_report') or "")

        if st.form_submit_button("💾 Save Doctor Recommendation", type="primary", use_container_width=True):
            if len(doctor_recommended) > 5:
                st.warning("⚠️ Please select at most 5 suspected pathogens.")
            else:
                payload = {
                    'prediction_id': rec['_id'],
                    'doctor_recommended_viruses': doctor_recommended,
                    'lab_id': lab_id.strip(), 'test_performed': test_performed,
                    'date_of_sample_collection': date_sample.strip(), 'sample_type': sample_type.strip(),
                    'diagnostic_method': diagnostic_method.strip(), 'laboratory_results': lab_results,
                    'confirmed_pathogen': confirmed, 'date_of_report': date_report.strip(),
                }
                if save_doctor_lab_data_to_db(payload):
                    _flash("✅ Doctor recommendation saved — case marked completed.")
                    _close_action()
                    st.rerun()
                else:
                    st.error("❌ Could not save the doctor recommendation.")


# ---- Delete (soft) ---------------------------------------------------------
def _render_delete_confirm(rec):
    st.subheader("🗑️ Delete record (soft delete)")
    st.warning(f"This hides **{rec.get('patient_name') or 'this patient'}** "
               f"(Study ID: {rec.get('patient_study_id') or '—'}) from the dashboard and "
               f"records list. The data is kept in the database and can be restored by an admin.")
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Confirm delete", type="primary", use_container_width=True, key=f"confirm_del_{rec['_id']}"):
        if soft_delete_record_in_db(rec['_id']):
            _flash("🗑️ Record deleted (soft).")
            _close_action()
            st.rerun()
        else:
            st.error("❌ Could not delete the record.")
    c2.button("Cancel", use_container_width=True, on_click=_close_action, key=f"cancel_del_{rec['_id']}")
