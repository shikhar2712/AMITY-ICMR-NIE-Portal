"""
Data Handler Module
Handles all database operations and data persistence
"""
from datetime import datetime
import pandas as pd
import logging
from database import get_db, test_db_connection
from typing import Dict, List, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

class DataHandler:
    """Handles all database operations for the virus prediction app"""
    
    def __init__(self):
        self.db = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database connection"""
        try:
            self.db = get_db()
            if self.db is not None:
                # Create indexes for better performance
                self._create_indexes()
                logger.info("DataHandler initialized successfully")
            else:
                logger.warning("Failed to initialize database connection")
        except Exception as e:
            logger.error(f"Error initializing DataHandler: {e}")
    
    def _get_next_patient_id(self) -> str:
        """Generate auto-incrementing patient ID (P001, P002, etc.)"""
        try:
            if self.db is None:
                return "P001"
            
            # Get the counter collection for patient IDs
            counters = self.db['counters']
            
            # Find and increment the patient counter
            result = counters.find_one_and_update(
                {'_id': 'patient_id'},
                {'$inc': {'sequence_value': 1}},
                upsert=True,
                return_document=True
            )
            
            # Format as P001, P002, etc.
            sequence_num = result.get('sequence_value', 1)
            return f"P{sequence_num:03d}"
            
        except Exception as e:
            logger.error(f"Error generating patient ID: {e}")
            # Fallback to timestamp-based ID
            import time
            return f"P{int(time.time())}"
    
    def _create_indexes(self):
        """Create database indexes for better performance"""
        try:
            if self.db is None:
                return
                
            # Create indexes on frequently queried fields
            collections = {
                'predictions': [
                    ('timestamp', -1),
                    ('patient_id', 1),
                    ('predicted_virus', 1),
                    ('validation.validated', 1),
                    ('validation.actual_virus_name', 1)
                ],
                'patients': [
                    ('patient_id', 1),
                    ('created_at', -1)
                ],
                'usage_stats': [
                    ('date', -1),
                    ('prediction_count', 1)
                ]
            }
            
            for collection_name, indexes in collections.items():
                collection = self.db[collection_name]
                for index_fields in indexes:
                    try:
                        collection.create_index([index_fields])
                    except Exception as e:
                        logger.warning(f"Index creation warning for {collection_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def save_prediction(self, 
                       patient_data: Dict, 
                       prediction_result: Dict,
                       model_info: Dict = None,
                       state_name: str = None,
                       district_name: str = None) -> Optional[str]:
        """
        Save prediction result to single collection with human-readable values
        
        Args:
            patient_data: Patient information and symptoms (encoded values)
            prediction_result: Model prediction results
            model_info: Model version and metadata
            state_name: Human-readable state name
            district_name: Human-readable district name
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            if self.db is None:
                logger.error("Database not initialized")
                return None
            
            # Use single collection for all data
            collection = self.db['virus_predictions']
            
            # Generate unique patient ID
            patient_id = self._get_next_patient_id()
            
            # Transform patient data to human-readable format
            readable_patient_info = {
                'patient_id': patient_id,
                # Keep core patient administrative fields first (matching app input order)
                'date_of_collection': patient_data.get('date_of_collection', ''),
                'patient_study_id': patient_data.get('patient_study_id', ''),
                'patient_mrd_id': patient_data.get('patient_mrd_id', ''),
                'hospital': patient_data.get('hospital', ''),
                'department': patient_data.get('department', ''),
                'date_of_admission': patient_data.get('date_of_admission', ''),
                'patient_name': patient_data.get('patient_name', ''),
                'address_line': patient_data.get('address_line', ''),
                'mobile_no': patient_data.get('mobile_no', ''),

                # Remaining demographics and model-relevant fields
                'age': patient_data.get('age'),
                'sex': 'Male' if patient_data.get('SEX') == 1 else 'Female',
                'patient_type': 'Inpatient' if patient_data.get('PATIENTTYPE') == 1 else 'Outpatient',
                'onset_of_illness': patient_data.get('onset_of_illness', ''),
                'duration_of_illness_days': patient_data.get('durationofillness'),
                'state_name': state_name or 'Unknown',
                'district_name': district_name or 'Unknown',
                'subdistrict': patient_data.get('subdistrict', ''),
                'pin_code': patient_data.get('pin_code', ''),
                'syndrome_name': patient_data.get('syndrome_name', ''),
                'syndrome_specification': patient_data.get('other_syndrome_specification', ''),
                'month_name': self._get_month_name(patient_data.get('month', 1)),
                'year': patient_data.get('year')
            }
            
            # Transform symptoms to human-readable format (flat structure for CSV)
            symptoms_readable = self._transform_symptoms_to_readable(patient_data)
            
            # Transform prediction results to human-readable
            prediction_readable = {
                'predicted_virus_name': prediction_result.get('predicted_virus'),
                'prediction_confidence_percent': prediction_result.get('confidence'),
                'top_1_virus': prediction_result.get('top_5_predictions', [{}])[0].get('virus', ''),
                'top_1_confidence': prediction_result.get('top_5_predictions', [{}])[0].get('confidence', 0),
                'top_2_virus': prediction_result.get('top_5_predictions', [{}])[1].get('virus', '') if len(prediction_result.get('top_5_predictions', [])) > 1 else '',
                'top_2_confidence': prediction_result.get('top_5_predictions', [{}])[1].get('confidence', 0) if len(prediction_result.get('top_5_predictions', [])) > 1 else 0,
                'top_3_virus': prediction_result.get('top_5_predictions', [{}])[2].get('virus', '') if len(prediction_result.get('top_5_predictions', [])) > 2 else '',
                'top_3_confidence': prediction_result.get('top_5_predictions', [{}])[2].get('confidence', 0) if len(prediction_result.get('top_5_predictions', [])) > 2 else 0,
                'top_4_virus': prediction_result.get('top_5_predictions', [{}])[3].get('virus', '') if len(prediction_result.get('top_5_predictions', [])) > 3 else '',
                'top_4_confidence': prediction_result.get('top_5_predictions', [{}])[3].get('confidence', 0) if len(prediction_result.get('top_5_predictions', [])) > 3 else 0,
                'top_5_virus': prediction_result.get('top_5_predictions', [{}])[4].get('virus', '') if len(prediction_result.get('top_5_predictions', [])) > 4 else '',
                'top_5_confidence': prediction_result.get('top_5_predictions', [{}])[4].get('confidence', 0) if len(prediction_result.get('top_5_predictions', [])) > 4 else 0
            }
            
            # Prepare complete document for single collection
            document = {
                # Patient information (flat structure)
                **readable_patient_info,
                
                # Symptoms (flat structure - each symptom as separate field)
                **symptoms_readable,
                
                # Predictions (flat structure)
                **prediction_readable,
                
                # Validation fields (empty initially, filled when validated)
                'validation_status': 'pending',
                'actual_virus_name': '',
                'actual_virus_category': '',
                'validation_confidence_level': '',
                'validation_notes': '',
                'validated_at': None,
                'validated_by': '',
                
                # Metadata
                'prediction_timestamp': datetime.utcnow(),
                'model_primary': model_info.get('model1', '') if model_info else '',
                'model_secondary': model_info.get('model2', '') if model_info else '',
                'app_version': '2.0'
            }
            
            # Insert document
            result = collection.insert_one(document)
            
            # Update usage statistics
            self._update_usage_stats()
            
            logger.info(f"Prediction saved with Patient ID: {patient_id}, Document ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return None
    
    
    def _get_month_name(self, month_num: int) -> str:
        """Convert month number to month name"""
        months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
        return months[month_num] if 1 <= month_num <= 12 else 'Unknown'
    
    def _transform_symptoms_to_readable(self, patient_data: Dict) -> Dict:
        """Transform symptom flags to human-readable flat structure for CSV export"""
        # Define all possible symptoms (from SYMPTOM_GROUPS in app.py)
        all_symptoms = [
            'HEADACHE', 'IRRITABILITY', 'ALTERED SENSORIUM', 'SOMNOLENCE', 'NECK RIGIDITY', 'SEIZURES',
            'DIARRHEA', 'DYSENTERY', 'NAUSEA', 'VOMITING', 'ABDOMINAL PAIN',
            'MALAISE', 'MYALGIA', 'ARTHRALGIA', 'CHILLS', 'RIGORS', 'FEVER',
            'BREATHLESSNESS', 'COUGH', 'RHINORRHEA', 'SORE THROAT',
            'BULLAE', 'PAPULAR RASH', 'PUSTULAR RASH', 'MUSCULAR RASH', 'MACULOPAPULAR RASH', 'ESCHAR',
            'DARK URINE', 'HEPATOMEGALY', 'JAUNDICE',
            'RED EYE', 'DISCHARGE EYES', 'CRUSHING EYES'
        ]
        
        # Create flat symptom structure
        symptoms_dict = {}
        for symptom in all_symptoms:
            readable_name = f"symptom_{symptom.lower().replace(' ', '_')}"
            symptoms_dict[readable_name] = 'Yes' if patient_data.get(symptom, 0) == 1 else 'No'
        
        return symptoms_dict
    
    def save_patient(self, patient_data: Dict) -> Optional[str]:
        """
        Save patient information to database
        
        Args:
            patient_data: Patient demographic and clinical information
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            if self.db is None:
                logger.error("Database not initialized")
                return None
            
            collection = self.db['patients']
            
            # Add metadata
            document = {
                **patient_data,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            
            result = collection.insert_one(document)
            logger.info(f"Patient saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving patient: {e}")
            return None
    
    def get_prediction_history(self, 
                              limit: int = 100,
                              patient_id: str = None) -> List[Dict]:
        """
        Retrieve prediction history
        
        Args:
            limit: Maximum number of records to return
            patient_id: Filter by specific patient ID
            
        Returns:
            List of prediction records
        """
        try:
            if self.db is None:
                return []
            
            collection = self.db['predictions']
            
            # Build query
            query = {}
            if patient_id:
                query['patient_data.patient_id'] = patient_id
            
            # Get records
            cursor = collection.find(query).sort('timestamp', -1).limit(limit)
            records = list(cursor)
            
            # Convert ObjectId to string for JSON serialization
            for record in records:
                record['_id'] = str(record['_id'])
                
            return records
            
        except Exception as e:
            logger.error(f"Error retrieving prediction history: {e}")
            return []
    
    def get_usage_statistics(self) -> Dict:
        """
        Get usage statistics
        
        Returns:
            Dictionary with usage statistics
        """
        try:
            if self.db is None:
                return {}
            
            predictions_collection = self.db['predictions']
            
            # Get total predictions
            total_predictions = predictions_collection.count_documents({})
            
            # Get predictions by virus type
            pipeline = [
                {
                    '$group': {
                        '_id': '$prediction_result.predicted_virus',
                        'count': {'$sum': 1}
                    }
                },
                {'$sort': {'count': -1}}
            ]
            
            virus_stats = list(predictions_collection.aggregate(pipeline))
            
            # Get predictions by date (last 30 days)
            from datetime import timedelta
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            daily_pipeline = [
                {
                    '$match': {
                        'timestamp': {'$gte': thirty_days_ago}
                    }
                },
                {
                    '$group': {
                        '_id': {
                            '$dateToString': {
                                'format': '%Y-%m-%d',
                                'date': '$timestamp'
                            }
                        },
                        'count': {'$sum': 1}
                    }
                },
                {'$sort': {'_id': 1}}
            ]
            
            daily_stats = list(predictions_collection.aggregate(daily_pipeline))
            
            return {
                'total_predictions': total_predictions,
                'virus_distribution': virus_stats,
                'daily_predictions': daily_stats,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting usage statistics: {e}")
            return {}
    
    def _update_usage_stats(self):
        """Update daily usage statistics"""
        try:
            if self.db is None:
                return
            
            collection = self.db['usage_stats']
            today = datetime.utcnow().date().isoformat()
            
            # Update or create today's stats
            collection.update_one(
                {'date': today},
                {
                    '$inc': {'prediction_count': 1},
                    '$set': {'last_updated': datetime.utcnow()}
                },
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error updating usage stats: {e}")
    
    def save_validation(self, validation_data: Dict) -> Optional[str]:
        """
        Save medical validation data within the same collection document
        
        Args:
            validation_data: Validation information including actual diagnosis
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            if self.db is None:
                logger.error("Database not initialized")
                return None
            
            # Use the single collection
            collection = self.db['virus_predictions']
            prediction_id = validation_data.get('prediction_id')
            
            if not prediction_id:
                logger.error("No prediction_id provided in validation data")
                return None
            
            # Convert string ID to ObjectId for MongoDB
            from bson import ObjectId
            try:
                object_id = ObjectId(prediction_id)
            except Exception as e:
                logger.error(f"Invalid prediction_id format: {e}")
                return None
            
            # Prepare validation fields for flat structure
            validation_fields = {
                'validation_status': 'validated',
                'actual_virus_name': validation_data.get('actual_virus_name', ''),
                'actual_virus_category': 'Main' if validation_data.get('actual_virus_key', '').startswith('main_') else 'Other',
                'validation_confidence_level': validation_data.get('confidence_level', ''),
                'validation_notes': validation_data.get('notes', ''),
                'validated_at': datetime.utcnow(),
                'validated_by': 'Medical Professional',
                'validation_accuracy': 'Correct' if validation_data.get('actual_virus_name') == validation_data.get('predicted_virus') else 'Incorrect'
            }
            
            # Update the document with validation data
            result = collection.update_one(
                {'_id': object_id},
                {
                    '$set': validation_fields,
                    '$currentDate': {'last_updated': True}
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Validation added to prediction ID: {prediction_id}")
                return prediction_id
            else:
                logger.warning(f"No prediction found with ID: {prediction_id}")
                return None
            
        except Exception as e:
            logger.error(f"Error saving validation: {e}")
            return None
    
    def get_validation_stats(self) -> Dict:
        """
        Get validation statistics for data collection and analysis
        Note: This is for research/improvement purposes, not system accuracy calculation
        
        Returns:
            Dictionary containing validation collection statistics
        """
        try:
            if self.db is None:
                return {'status': 'error', 'message': 'Database not initialized'}
            
            collection = self.db['predictions']
            
            # Count total predictions with validation data
            total_validations = collection.count_documents({'validation.validated': True})
            
            # Get validation distribution by actual virus
            pipeline = [
                {
                    '$match': {'validation.validated': True}
                },
                {
                    '$group': {
                        '_id': '$validation.actual_virus_name',
                        'count': {'$sum': 1}
                    }
                },
                {'$sort': {'count': -1}}
            ]
            
            validation_distribution = list(collection.aggregate(pipeline))
            
            # Get validation confidence distribution
            confidence_pipeline = [
                {
                    '$match': {'validation.validated': True}
                },
                {
                    '$group': {
                        '_id': '$validation.confidence_level',
                        'count': {'$sum': 1}
                    }
                }
            ]
            
            confidence_stats = list(collection.aggregate(confidence_pipeline))
            
            return {
                'status': 'success',
                'total_validations': total_validations,
                'validation_distribution': validation_distribution,
                'confidence_distribution': confidence_stats,
                'last_updated': datetime.utcnow().isoformat(),
                'note': 'This data is for research and model improvement purposes'
            }
            
        except Exception as e:
            logger.error(f"Error getting validation stats: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def export_to_csv(self, limit: int = None) -> Optional[pd.DataFrame]:
        """
        Export all data to CSV-ready DataFrame format
        
        Args:
            limit: Maximum number of records to export (None for all)
            
        Returns:
            pandas DataFrame ready for CSV export
        """
        try:
            if self.db is None:
                logger.error("Database not initialized")
                return None
            
            collection = self.db['virus_predictions']
            
            # Build query to exclude MongoDB internal fields
            projection = {'_id': 0, 'encoded_data': 0}  # Exclude internal fields
            
            # Get records
            if limit:
                cursor = collection.find({}, projection).sort('prediction_timestamp', -1).limit(limit)
            else:
                cursor = collection.find({}, projection).sort('prediction_timestamp', -1)
            
            records = list(cursor)
            
            if not records:
                logger.warning("No records found for export")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Format timestamps for better readability
            if 'prediction_timestamp' in df.columns:
                df['prediction_timestamp'] = df['prediction_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            if 'validated_at' in df.columns:
                df['validated_at'] = df['validated_at'].apply(
                    lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else ''
                )
            
            # Reorder columns for better CSV structure
            column_order = [
                'patient_id', 'age', 'sex', 'patient_type',
                'state_name', 'district_name', 'syndrome_name', 'syndrome_specification',
                'onset_of_illness', 'duration_of_illness_days', 'month_name', 'year', 'prediction_timestamp'
            ]
            
            # Add symptom columns
            symptom_cols = [col for col in df.columns if col.startswith('symptom_')]
            column_order.extend(sorted(symptom_cols))
            
            # Add prediction columns
            prediction_cols = [
                'predicted_virus_name', 'prediction_confidence_percent',
                'top_1_virus', 'top_1_confidence', 'top_2_virus', 'top_2_confidence',
                'top_3_virus', 'top_3_confidence', 'top_4_virus', 'top_4_confidence',
                'top_5_virus', 'top_5_confidence'
            ]
            column_order.extend(prediction_cols)
            
            # Add validation columns
            validation_cols = [
                'validation_status', 'actual_virus_name', 'actual_virus_category',
                'validation_confidence_level', 'validation_notes', 'validated_at',
                'validated_by', 'validation_accuracy'
            ]
            column_order.extend(validation_cols)
            
            # Add metadata columns
            metadata_cols = ['model_primary', 'model_secondary', 'app_version']
            column_order.extend(metadata_cols)
            
            # Reorder DataFrame columns
            existing_cols = [col for col in column_order if col in df.columns]
            remaining_cols = [col for col in df.columns if col not in existing_cols]
            final_column_order = existing_cols + remaining_cols
            
            df = df[final_column_order]
            
            logger.info(f"Exported {len(df)} records to DataFrame")
            return df
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return None
    
    def health_check(self) -> Dict:
        """
        Perform health check on database connection and operations
        
        Returns:
            Health check results
        """
        try:
            # Check if we have a database instance (connection already established)
            if self.db is None:
                return {
                    'status': 'error',
                    'message': 'Database instance not available',
                    'details': {
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }
            
            # Test basic operations using existing connection
            try:
                # Simple test - count documents in virus_predictions collection
                predictions_count = self.db['virus_predictions'].count_documents({})
                
                return {
                    'status': 'healthy',
                    'message': 'All database operations working',
                    'details': {
                        'connection': 'OK',
                        'total_predictions': predictions_count,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }
                
            except Exception as op_error:
                return {
                    'status': 'error',
                    'message': f'Database operations failed: {str(op_error)}',
                    'details': {
                        'error': str(op_error),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Health check failed: {str(e)}',
                'details': {
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }

# Global data handler instance
data_handler = DataHandler()

# Convenience functions for use in app.py
def save_prediction_to_db(patient_data: Dict, 
                         prediction_result: Dict, 
                         model_info: Dict = None,
                         state_name: str = None,
                         district_name: str = None) -> Optional[str]:
    """Save prediction to database"""
    return data_handler.save_prediction(patient_data, prediction_result, model_info, state_name, district_name)

def save_validation_to_db(validation_data: Dict) -> Optional[str]:
    """Save validation to database"""
    return data_handler.save_validation(validation_data)

def get_db_health() -> Dict:
    """Get database health status"""
    return data_handler.health_check()

def get_prediction_stats() -> Dict:
    """Get prediction usage statistics"""
    return data_handler.get_usage_statistics()

def get_validation_stats() -> Dict:
    """Get validation statistics"""
    return data_handler.get_validation_stats()

def export_data_to_csv(limit: int = None) -> Optional[pd.DataFrame]:
    """Export data to CSV-ready DataFrame"""
    return data_handler.export_to_csv(limit)