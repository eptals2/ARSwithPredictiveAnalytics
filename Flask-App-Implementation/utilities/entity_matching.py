from typing import Dict, List, Set, Tuple, Any
import re
import xgboost as xgb
import logging

class EntityMatcher:
    def __init__(self, model_path: str, xgboost_model_path: str = None):
        """Initialize the EntityMatcher with model paths"""
        self.model_path = model_path
        self.xgboost_model_path = xgboost_model_path
        self.core_skills = {
            'python', 'java', 'javascript', 'c++', 'sql', 'html', 'css', 
            'web development', 'programming', 'software development'
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text using regex patterns until RoBERTa model is available
        """
        text = text.lower()
        
        # Initialize entities
        entities = {
            'AGE': [],
            'GENDER': [],
            'ADDRESS': [],
            'SKILLS': [],
            'EXPERIENCE': [],
            'EDUCATION': [],
            'CERTIFICATION': []
        }
        
        # Extract age
        age_pattern = r'\b(\d{1,2})\s*(?:years?\s*(?:old|of\s*age)|yrs?)\b'
        age_match = re.search(age_pattern, text)
        if age_match:
            entities['AGE'].append(age_match.group(1))
            
        # Extract gender
        gender_words = {'male', 'female', 'm', 'f'}
        for word in text.split():
            if word in gender_words:
                entities['GENDER'].append(word)
                break
                
        # Extract skills
        skill_patterns = {
            'python': r'\bpython\b',
            'java': r'\bjava\b',
            'javascript': r'\bjavascript\b|\bjs\b',
            'c++': r'\bc\+\+\b',
            'sql': r'\bsql\b',
            'html': r'\bhtml\b',
            'css': r'\bcss\b',
            'web development': r'\bweb\s*development\b',
            'programming': r'\bprogramming\b',
            'software development': r'\bsoftware\s*development\b'
        }
        
        for skill, pattern in skill_patterns.items():
            if re.search(pattern, text):
                entities['SKILLS'].append(skill)
                
        # Extract education
        education_patterns = {
            'computer science': r'\bcomputer\s*science\b|\bcs\b',
            'information technology': r'\binformation\s*technology\b|\bit\b',
            'software engineering': r'\bsoftware\s*engineering\b',
            'computer engineering': r'\bcomputer\s*engineering\b'
        }
        
        for edu, pattern in education_patterns.items():
            if re.search(pattern, text):
                entities['EDUCATION'].append(edu)
                
        # Extract experience
        exp_pattern = r'\b(\d+)\s*(?:years?\s*experience|yrs?\s*exp)\b'
        exp_match = re.search(exp_pattern, text)
        if exp_match:
            entities['EXPERIENCE'].append(f"{exp_match.group(1)} years")
            
        # Extract certifications
        cert_patterns = {
            'microsoft': r'\bmicrosoft\s*certified\b|\bmcp\b|\bmcsa\b|\bmcse\b',
            'aws': r'\baws\s*certified\b',
            'oracle': r'\boracle\s*certified\b|\boca\b|\bocp\b',
            'cisco': r'\bcisco\s*certified\b|\bccna\b|\bccnp\b',
            'comptia': r'\bcomptia\b|\ba\+\b|\bnetwork\+\b|\bsecurity\+\b'
        }
        
        for cert, pattern in cert_patterns.items():
            if re.search(pattern, text):
                entities['CERTIFICATION'].append(cert)
                
        return entities

    def analyze_resume(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        """
        Analyze resume against job requirements
        """
        try:
            # Extract entities
            resume_entities = self.extract_entities(resume_text)
            job_entities = self.extract_entities(job_text)
            
            # Initialize entity analysis
            entity_analysis = {}
            similarity_scores = {}
            
            # Analyze each entity type
            for entity_type in resume_entities.keys():
                entity_analysis[entity_type] = {
                    'requirements': job_entities[entity_type],
                    'candidate': resume_entities[entity_type],
                    'matching': [],
                    'missing': [],
                    'matching_score': 0.0
                }
                
                # Calculate matches and scores
                if entity_type in ['SKILLS', 'EDUCATION', 'CERTIFICATION']:
                    matches = set(job_entities[entity_type]) & set(resume_entities[entity_type])
                    missing = set(job_entities[entity_type]) - set(resume_entities[entity_type])
                    
                    entity_analysis[entity_type]['matching'] = list(matches)
                    entity_analysis[entity_type]['missing'] = list(missing)
                    
                    # Calculate Jaccard similarity (intersection over union)
                    union = set(job_entities[entity_type]) | set(resume_entities[entity_type])
                    if union:
                        score = len(matches) / len(union)
                    else:
                        score = 0.0
                        
                    entity_analysis[entity_type]['matching_score'] = score * 100  # Convert to percentage
                    similarity_scores[entity_type.lower()] = score
                
                # Simple match for scalar entities
                else:
                    if job_entities[entity_type] and resume_entities[entity_type]:
                        is_match = job_entities[entity_type][0].lower() == resume_entities[entity_type][0].lower()
                        entity_analysis[entity_type]['matching'] = resume_entities[entity_type] if is_match else []
                        entity_analysis[entity_type]['missing'] = job_entities[entity_type] if not is_match else []
                        entity_analysis[entity_type]['matching_score'] = 100.0 if is_match else 0.0
                        similarity_scores[entity_type.lower()] = 1.0 if is_match else 0.0
                    else:
                        similarity_scores[entity_type.lower()] = 0.0
            
            # Calculate overall match score
            overall_match = self._calculate_overall_match(
                entity_analysis=entity_analysis,
                similarity_scores=similarity_scores,
                role_confidence=100.0  # Default to 100% since we're not using role confidence
            )
            
            # Get suitability status
            suitability_status = self._get_suitability_status(overall_match)
            
            # Get score breakdown
            score_breakdown = {
                'skills': round(similarity_scores['skills'] * 100, 1),
                'experience': round(similarity_scores['experience'] * 100, 1),
                'education': round(similarity_scores['education'] * 100, 1),
                'certification': round(similarity_scores['certification'] * 100, 1)
            }
            
            # Get XGBoost prediction
            xgboost_result = self.predict_role_with_xgboost(entity_analysis, similarity_scores)
            
            return {
                'entity_analysis': entity_analysis,
                'overall_match': overall_match,
                'score_breakdown': score_breakdown,
                'suitability_status': suitability_status,
                'role_confidence': xgboost_result.get('confidence', 0.0)  # Add XGBoost confidence score
            }
            
        except Exception as e:
            print(f"Error in analyze_resume: {str(e)}")
            return {
                'overall_match': 0.0,
                'suitability_status': 'Not Suitable'
            }

    def _calculate_overall_match(self, entity_analysis: Dict[str, Dict[str, object]], similarity_scores: Dict[str, float], role_confidence: float) -> float:
        """
        Calculate overall match score based on entity analysis and role confidence
        """
        # Weights for each entity type
        weights = {
            'skills': 0.4,
            'experience': 0.3,
            'education': 0.2,
            'certification': 0.1
        }
        
        # Calculate weighted score
        score = sum(
            similarity_scores[entity_type] * weights[entity_type] 
            for entity_type in weights.keys()
        )
        
        # Apply role confidence (already in percentage)
        score = (score + (role_confidence / 100)) / 2
        
        # Convert to percentage and round
        return round(score * 100, 1)

    def predict_role_with_xgboost(self, entity_analysis: Dict[str, Dict[str, object]], similarity_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict suitable role or recommend alternative roles based on entity analysis
        """
        try:
            # Calculate overall score
            overall_score = (
                similarity_scores.get('skills', 0) * 0.4 +
                similarity_scores.get('experience', 0) * 0.3 +
                similarity_scores.get('education', 0) * 0.2 +
                similarity_scores.get('certification', 0) * 0.1
            ) * 100  # Convert to percentage

            # Apply penalties
            if not self._has_cs_it_education(entity_analysis['EDUCATION']['candidate']):
                overall_score *= 0.5
            if not self._has_relevant_experience(entity_analysis['EXPERIENCE']['candidate']):
                overall_score *= 0.8

            if overall_score >= 50:
                # Use XGBoost for high-potential candidates
                features = {
                    'skills_similarity': similarity_scores.get('skills', 0),
                    'experience_similarity': similarity_scores.get('experience', 0),
                    'education_similarity': similarity_scores.get('education', 0),
                    'certification_similarity': similarity_scores.get('certification', 0),
                    'has_cs_education': self._has_cs_it_education(entity_analysis['EDUCATION']['candidate']),
                    'has_relevant_experience': self._has_relevant_experience(entity_analysis['EXPERIENCE']['candidate']),
                    'core_skills_coverage': self._calculate_core_skills_coverage(entity_analysis['SKILLS']['candidate']),
                    'years_experience': self._extract_years_experience(entity_analysis['EXPERIENCE']['candidate'])
                }
                
                # Convert to DMatrix format
                feature_names = list(features.keys())
                feature_values = [features[name] for name in feature_names]
                feature_matrix = xgb.DMatrix([feature_values], feature_names=feature_names)
                
                try:
                    # Load and predict with model
                    model = xgb.load_model(self.xgboost_model_path)
                    prediction = model.predict(feature_matrix)
                    
                    # Get role and confidence
                    role = 'Programmer' if prediction[0] >= 0.5 else 'Encoder'
                    confidence = float(prediction[0] if prediction[0] >= 0.5 else 1 - prediction[0])
                    
                    return {
                        'role': role,
                        'confidence': round(confidence * 100, 1),
                        'recommendation': None
                    }
                except Exception as e:
                    logging.warning(f"XGBoost model error: {e}. Using rule-based prediction.")
                    role = 'Programmer' if overall_score >= 70 else 'Encoder'
                    return {
                        'role': role,
                        'confidence': round(overall_score, 1),
                        'recommendation': None
                    }
            else:
                # Provide recommendations based on skills and experience
                recommendations = self._get_role_recommendations(
                    skills=entity_analysis['SKILLS']['candidate'],
                    experience=entity_analysis['EXPERIENCE']['candidate'],
                    education=entity_analysis['EDUCATION']['candidate']
                )
                
                return {
                    'role': 'Not Suitable',
                    'confidence': round(overall_score, 1),
                    'recommendation': recommendations
                }
            
        except Exception as e:
            logging.error(f"Error in role prediction: {e}")
            return {
                'role': 'Unknown',
                'confidence': 0.0,
                'recommendation': None
            }

    def _calculate_core_skills_coverage(self, candidate_skills: List[str]) -> float:
        """Calculate coverage of core programming skills"""
        core_skills = {
            'python', 'java', 'javascript', 'c++', 'sql', 
            'html', 'css', 'web development', 'programming', 'software development'
        }
        candidate_skills_lower = {skill.lower() for skill in candidate_skills}
        matched_skills = core_skills.intersection(candidate_skills_lower)
        return len(matched_skills) / len(core_skills)

    def _extract_years_experience(self, experience_list: List[str]) -> float:
        """Extract years of experience from experience descriptions"""
        years = 0.0
        for exp in experience_list:
            # Look for patterns like "X years" or "X+ years"
            matches = re.findall(r'(\d+)(?:\+)?\s*years?', exp.lower())
            if matches:
                years = max(years, float(matches[0]))
        return years

    def _has_cs_it_education(self, education_list: List[str]) -> bool:
        """Check if candidate has CS/IT education"""
        keywords = {'computer science', 'information technology', 'software engineering'}
        education_text = ' '.join(education_list).lower()
        return any(keyword in education_text for keyword in keywords)

    def _has_relevant_experience(self, experience_list: List[str]) -> bool:
        """Check if candidate has relevant software development experience"""
        keywords = {'software', 'programming', 'developer', 'web development'}
        experience_text = ' '.join(experience_list).lower()
        return any(keyword in experience_text for keyword in keywords)

    def _get_role_recommendations(self, skills: List[str], experience: List[str], education: List[str]) -> Dict[str, Any]:
        """
        Generate role recommendations based on candidate's profile
        """
        # Define role requirements
        role_requirements = {
            'Web Developer': {
                'skills': {'html', 'css', 'javascript', 'web development', 'react', 'angular', 'vue', 'php'},
                'weight': 0.4
            },
            'Data Analyst': {
                'skills': {'python', 'sql', 'data analysis', 'excel', 'statistics', 'tableau', 'power bi'},
                'weight': 0.35
            },
            'QA Engineer': {
                'skills': {'testing', 'selenium', 'automation', 'quality assurance', 'jira', 'test cases'},
                'weight': 0.3
            },
            'IT Support': {
                'skills': {'troubleshooting', 'networking', 'hardware', 'customer service', 'windows', 'linux'},
                'weight': 0.25
            }
        }

        # Calculate match scores for each role
        role_scores = {}
        candidate_skills = {skill.lower() for skill in skills}
        
        for role, requirements in role_requirements.items():
            required_skills = requirements['skills']
            matching_skills = candidate_skills.intersection(required_skills)
            
            # Calculate weighted score
            score = len(matching_skills) / len(required_skills) * requirements['weight'] * 100
            
            # Boost score if relevant experience exists
            if any(exp.lower() in ' '.join(experience).lower() for exp in required_skills):
                score *= 1.2
            
            # Boost score if relevant education exists
            if any(edu.lower() in ' '.join(education).lower() for edu in ['computer', 'it', 'software', 'engineering']):
                score *= 1.1
            
            role_scores[role] = min(round(score, 1), 100)

        # Sort roles by score and get top 2
        sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        
        return {
            'suggested_roles': [
                {
                    'title': role,
                    'match_score': score,
                    'required_skills': list(role_requirements[role]['skills'])
                }
                for role, score in sorted_roles if score > 30
            ]
        }

    def _get_suitability_status(self, overall_score: float) -> str:
        """Determine candidate suitability based on overall score"""
        if overall_score >= 80:
            return "Highly Suitable"
        elif overall_score >= 60:
            return "Suitable"
        elif overall_score >= 40:
            return "Moderately Suitable"
        else:
            return "Not Suitable"
