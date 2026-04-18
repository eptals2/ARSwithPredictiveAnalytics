from typing import Dict, List, Set, Tuple, Any
import re
import xgboost as xgb
import nltk
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import defaultdict

nltk.download('punkt', quiet=True)

class EntityMatcher:
    def __init__(self, model_path: str, xgboost_model_path: str = None):
        """Initialize the EntityMatcher with model paths"""
        self.model_path = model_path
        self.xgboost_model_path = xgboost_model_path
        
        # Initialize NER model
        try:
            # Always use the pre-trained model for reliability
            logging.info("Using pre-trained BERT-NER model")
            self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
            
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
            logging.info("NER pipeline initialized successfully")
        except Exception as e:
            logging.error("Error loading NER model: %s", str(e))
            self.ner_pipeline = None
            
        # Entity type mapping for NER labels
        self.entity_type_mapping = {
            'AGE': ['AGE'],
            'GENDER': ['GENDER'],
            'ADDRESS': ['LOC', 'GPE', 'LOCATION'],
            'SKILLS': ['SKILL', 'TECHNOLOGY'],
            'EXPERIENCE': ['EXPERIENCE', 'WORK_YEARS'],
            'EDUCATION': ['EDUCATION', 'DEGREE'],
            'CERTIFICATION': ['CERTIFICATION', 'CERT']
        }
        
        # Weights for different match types
        self.match_weights = {
            'exact': 1.0,
            'variation': 0.8,
            'partial': 0.5,
            'ngram': 0.3
        }
        
        # Initialize skill variations mapping
        self.skill_mapping = {}
        for main_skill, variations in self.skill_variations.items():
            for var in variations:
                self.skill_mapping[var] = main_skill
            self.skill_mapping[main_skill] = main_skill

    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        # Standardize whitespace
        text = ' '.join(text.split())
        return text

    def get_ngrams(self, text: str) -> Set[str]:
        """Generate unigrams and bigrams from text"""
        tokens = text.split()
        unigrams = set(tokens)
        bigrams = set(' '.join(bg) for bg in ngrams(tokens, 2))
        return unigrams | bigrams

    def calculate_enhanced_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate enhanced similarity score using weighted matching"""
        if not set1 or not set2:
            return 0.0
            
        total_score = 0.0
        matched_items = set()
        
        # Process exact matches first
        exact_matches = set1 & set2
        for match in exact_matches:
            total_score += self.match_weights['exact']
            matched_items.add(match)
        
        # Process variations
        remaining1 = set1 - matched_items
        remaining2 = set2 - matched_items
        for item1 in remaining1:
            normalized1 = self.normalize_text(item1)
            for item2 in remaining2:
                normalized2 = self.normalize_text(item2)
                
                # Check for variations
                if normalized1 in self.skill_mapping and normalized2 in self.skill_mapping:
                    if self.skill_mapping[normalized1] == self.skill_mapping[normalized2]:
                        total_score += self.match_weights['variation']
                        matched_items.add(item1)
                        break
                
                # Check for partial matches
                if normalized1 in normalized2 or normalized2 in normalized1:
                    total_score += self.match_weights['partial']
                    matched_items.add(item1)
                    break
                    
                # Check for ngram matches
                ngrams1 = self.get_ngrams(normalized1)
                ngrams2 = self.get_ngrams(normalized2)
                if ngrams1 & ngrams2:
                    total_score += self.match_weights['ngram']
                    matched_items.add(item1)
                    break
        
        # Normalize score based on the maximum possible score
        max_possible_score = max(len(set1), len(set2)) * self.match_weights['exact']
        return total_score / max_possible_score if max_possible_score > 0 else 0.0

    def extract_entities_with_ner(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using RoBERTa NER model"""
        entities = defaultdict(list)
        
        try:
            if self.ner_pipeline:
                # Run NER pipeline
                ner_results = self.ner_pipeline(text)
                
                # Process NER results
                for result in ner_results:
                    entity_text = result['word'].strip()
                    entity_type = result['entity_group']
                    score = result['score']
                    
                    # Skip low confidence predictions and empty strings
                    if score < 0.5 or not entity_text:
                        continue
                        
                    # Map NER labels to our entity types
                    for our_type, ner_labels in self.entity_type_mapping.items():
                        if entity_type in ner_labels:
                            if entity_text not in entities[our_type]:
                                entities[our_type].append(entity_text)
                            break
                
                # Post-process entities
                entities = self._post_process_entities(dict(entities))
                
                # Ensure all entity types exist
                for entity_type in ['AGE', 'GENDER', 'ADDRESS', 'SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION']:
                    if entity_type not in entities:
                        entities[entity_type] = []
                
                return dict(entities)
            else:
                # Fallback to regex-based extraction
                return self.extract_entities_with_regex(text)
        except Exception as e:
            logging.error("Error in NER extraction: %s", str(e))
            return self.extract_entities_with_regex(text)

    def _post_process_entities(self, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Post-process extracted entities for normalization and validation"""
        processed = defaultdict(list)
        
        # Ensure all entity types exist
        for entity_type in ['AGE', 'GENDER', 'ADDRESS', 'SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION']:
            if entity_type not in entities:
                entities[entity_type] = []
        
        # Normalize age
        for age in entities.get('AGE', []):
            if isinstance(age, str) and re.search(r'\d+', age):
                try:
                    age_value = int(re.search(r'\d+', age)[0])
                    if 15 <= age_value <= 100:  # Basic age validation
                        processed['AGE'].append(str(age_value))
                except (ValueError, TypeError):
                    continue
        
        # Normalize gender
        gender_mapping = {
            'male': 'male',
            'm': 'male',
            'female': 'female',
            'f': 'female'
        }
        for gender in entities.get('GENDER', []):
            gender_lower = gender.lower().strip()
            if gender_lower in gender_mapping:
                processed['GENDER'].append(gender_mapping[gender_lower])
            
        # Normalize address
        for addr in entities.get('ADDRESS', []):
            if isinstance(addr, str):
                addr_clean = addr.strip()
                if addr_clean and addr_clean not in processed['ADDRESS']:
                    processed['ADDRESS'].append(addr_clean)
                    
        # Normalize skills
        for skill in entities.get('SKILLS', []):
            if isinstance(skill, str):
                skill_lower = skill.lower().strip()
                # Check for variations
                if skill_lower in self.skill_mapping:
                    normalized_skill = self.skill_mapping[skill_lower]
                    if normalized_skill not in processed['SKILLS']:
                        processed['SKILLS'].append(normalized_skill)
                else:
                    # Check for partial matches in skill variations
                    found_match = False
                    for main_skill, variations in self.skill_variations.items():
                        if skill_lower in variations or any(var in skill_lower for var in variations):
                            if main_skill not in processed['SKILLS']:
                                processed['SKILLS'].append(main_skill)
                            found_match = True
                            break
                    if not found_match and skill_lower not in processed['SKILLS']:
                        processed['SKILLS'].append(skill_lower)
            
        # Normalize experience
        for exp in entities.get('EXPERIENCE', []):
            if isinstance(exp, str):
                exp_match = re.search(r'(\d+)(?:\+)?\s*years?', exp.lower())
                if exp_match:
                    years = exp_match.group(1)
                    if years not in processed['EXPERIENCE']:
                        processed['EXPERIENCE'].append(f"{years} years")
                        
        # Normalize education
        education_keywords = {
            'computer science', 'information technology', 'software engineering',
            'computer engineering', 'elementary', 'high school', 'vocational',
            'tech-voc', 'diploma', "bachelor's", 'masters', 'doctorate', 'phd'
            'data science', 'data analytics', 'artificial intelligence', 'machine learning', 
            'mathematics', 'statistics', 'biostatistics', 'economics', 'business administration',
            'finance', 'law', 'psychology', 'nursing', 'engineering', 'architecture', 'pharmacy',
            'education', 'history', 'literature', 'communication', 'marketing', 'senior high',
            'junior high', 'secondary', 'tertiary'

        }
        for edu in entities.get('EDUCATION', []):
            if isinstance(edu, str):
                edu_lower = edu.lower().strip()
                # Check for education keywords
                for keyword in education_keywords:
                    if keyword in edu_lower and edu_lower not in processed['EDUCATION']:
                        processed['EDUCATION'].append(edu_lower)
                        break
                        
        # Normalize certifications
        cert_keywords = {
            'microsoft': ['microsoft', 'mcp', 'mcsa', 'mcse'],
            'aws': ['aws', 'amazon'],
            'oracle': ['oracle', 'oca', 'ocp'],
            'cisco': ['cisco', 'ccna', 'ccnp'],
            'comptia': ['comptia', 'a+', 'network+', 'security+'],
            'pmp': ['pmp', 'project management professional'],
            'itil': ['itil'],
            'scrum': ['scrum', 'psm', 'csm'],
            'nc': ['nc', 'national certificate', 'tesda'],
            'nc-ii': ['nc-ii', 'national certificate ii', 'tesda ii'],
            'nc-i': ['nc-i', 'national certificate i', 'tesda i']
        }
        for cert in entities.get('CERTIFICATION', []):
            if isinstance(cert, str):
                cert_lower = cert.lower().strip()
                for cert_name, keywords in cert_keywords.items():
                    if any(keyword in cert_lower for keyword in keywords):
                        if cert_name not in processed['CERTIFICATION']:
                            processed['CERTIFICATION'].append(cert_name)
                        break
        
        return dict(processed)

    def extract_entities_with_regex(self, text: str) -> Dict[str, List[str]]:
        """Fallback regex-based entity extraction"""
        text = text.lower()
        
        # Initialize entities with empty lists
        entities = defaultdict(list)
        
        # Extract age
        age_pattern = r'\b(\d{1,2})\s*(?:years?\s*(?:old|of\s*age)|yrs?)\b'
        age_matches = re.finditer(age_pattern, text)
        for match in age_matches:
            entities['AGE'].append(match.group(1))
            
        # Extract gender
        gender_words = {'male', 'female', 'm', 'f'}
        for word in text.split():
            if word in gender_words:
                entities['GENDER'].append(word)
                
        # Extract address (basic patterns)
        address_patterns = [
            r'\b\d+\s+[a-z\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr)\b',
            r'\b[a-z\s]+(?:city|province|region|district)\b'
        ]
        for pattern in address_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities['ADDRESS'].append(match.group())

        # Extract skills (using existing patterns)
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
            'software development': r'\bsoftware\s*development\b',
            'project management': r'\bproject\s*management\b|\bpmp\b',
            'data science': r'\bdata\s*science\b|\bds\b',
            'machine learning': r'\bmachine\s*learning\b|\bml\b',
            'statistics': r'\bstatistics\b',
            'data analysis': r'\bdata\s*analysis\b',
            'data visualization': r'\bdata\s*visualization\b',
            'data mining': r'\bdata\s*mining\b',
            'deep learning': r'\bdeep\s*learning\b',
            'natural language processing': r'\bnatural\s*language\s*processing\b|\bnlp\b',
            'artificial intelligence': r'\bartificial\s*intelligence\b|\bai\b',
            'computer vision': r'\bcomputer\s*vision\b',
            'human resources': r'\bhuman\s*resources\b|\bhr\b',
            'customer service': r'\bcustomer\s*service\b',
            'marketing': r'\bmarketing\b',
            'communication': r'\bcommunication\b',
            'leadership': r'\bleadership\b',
            'team management': r'\bteam\s*management\b',
            'time management': r'\btime\s*management\b',
            'public speaking': r'\bpublic\s*speaking\b',
            'problem solving': r'\bproblem\s*solving\b',
            'teamwork': r'\bteamwork\b',
            'business development': r'\bbusiness\s*development\b',
            'operations': r'\boperations\b',
            'finance': r'\bfinance\b',
            'accounting': r'\baccounting\b',
            'economics': r'\beconomics\b',
            'business analysis': r'\bbusiness\s*analysis\b',
            'project planning': r'\bproject\s*planning\b',
            'project execution': r'\bproject\s*execution\b',
            'project monitoring': r'\bproject\s*monitoring\b',
            'project control': r'\bproject\s*control\b',
            'quality assurance': r'\bquality\s*assurance\b',
            'quality control': r'\bquality\s*control\b',
            'risk management': r'\brisk\s*management\b',
            'cost management': r'\bcost\s*management\b',
            'procurement': r'\bprocurement\b',
            'contract management': r'\bcontract\s*management\b',
            'stakeholder management': r'\bstakeholder\s*management\b',
            'communication planning': r'\bcommunication\s*planning\b',
            'resource allocation': r'\bresource\s*allocation\b',
            'resource management': r'\bresource\s*management\b',
            'accountancy': r'\baccountancy\b',
            'business administration': r'\bbusiness\s*administration\b',
            'computer information systems': r'\bcomputer\s*information\s*systems\b',
            'computer programming': r'\bcomputer\s*programming\b',
            'electronics and communications engineering': r'\belectronics\s*and\s*communications\s*engineering\b',
            'electrical engineering': r'\belectrical\s*engineering\b',
            'entrepreneurship': r'\bentrepreneurship\b',
            'hospitality management': r'\bhospitality\s*management\b',
            'human resource management': r'\bhuman\s*resource\s*management\b',
            'information systems': r'\binformation\s*systems\b',
            'information technology': r'\binformation\s*technology\b',
            'management': r'\bmanagement\b',
            'marketing management': r'\bmarketing\s*management\b',
            'mechanical engineering': r'\bmechanical\s*engineering\b',
            'nursing': r'\bnursing\b',
            'psychology': r'\bpsychology\b',
            'public relations': r'\bpublic\s*relations\b',
            'sales': r'\bsales\b',
            'social work': r'\bsocial\s*work\b',
            'strategic planning': r'\bstrategic\s*planning\b',
            'supply chain management': r'\bsupply\s*chain\s*management\b',
            'teaching': r'\bteaching\b',
            'training': r'\btraining\b',
            'writing': r'\bwriting\b',
            'network administrator': r'\bnetwork\s*administrator\b',
            'system administrator': r'\bsystem\s*administrator\b',
            'help desk': r'\bhelp\s*desk\b',
            'customer service representative': r'\bcustomer\s*service\s*representative\b',
            'sales representative': r'\bsales\s*representative\b',
            'accountant': r'\baccountant\b',
            'accounting clerk': r'\baccounting\s*clerk\b',
            'administrative assistant': r'\badministrative\s*assistant\b',
            'human resource assistant': r'\bhuman\s*resource\s*assistant\b',
            'marketing assistant': r'\bmarketing\s*assistant\b',
            'operations assistant': r'\boperations\s*assistant\b',
            'project assistant': r'\bproject\s*assistant\b',
            'sales assistant': r'\bsales\s*assistant\b'
        }
        
        for skill, pattern in skill_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                if skill not in entities['SKILLS']:
                    entities['SKILLS'].append(skill)
                
        # Extract education
        education_patterns = {
            'computer science': r'\bcomputer\s*science\b|\bcs\b',
            'information technology': r'\binformation\s*technology\b|\bit\b',
            'software engineering': r'\bsoftware\s*engineering\b',
            'computer engineering': r'\bcomputer\s*engineering\b',
            'elementary education': r'\belementary\s*education\b',
            'high school': r'\bhigh\s*school\b',
            'vocational': r'\bvocational\b',
            'tech-voc': r'\btech-voc\b',
            'diploma': r'\bdiploma\b',
            'bachelor': r'\bbachelor\'?s?\s*(?:degree)?\b|\bbs\b|\bba\b',
            'master': r'\bmaster\'?s?\s*(?:degree)?\b|\bms\b|\bma\b',
            'doctorate': r'\bdoctorate\b|\bphd\b'
        }
        
        for edu, pattern in education_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                if edu not in entities['EDUCATION']:
                    entities['EDUCATION'].append(edu)
                
        # Extract experience
        exp_patterns = [
            r'\b(\d+)\s*(?:years?\s*experience|yrs?\s*exp)\b',
            r'\bexperience\s*:\s*(\d+)\s*years?\b',
            r'\b(\d+)\s*years?\s*(?:of\s*)?work(?:ing)?\s*experience\b'
        ]
        for pattern in exp_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                exp = f"{match.group(1)} years"
                if exp not in entities['EXPERIENCE']:
                    entities['EXPERIENCE'].append(exp)
            
        # Extract certifications
        cert_patterns = {
            'microsoft': r'\bmicrosoft\s*certified\b|\bmcp\b|\bmcsa\b|\bmcse\b',
            'aws': r'\baws\s*certified\b|\baws\s*certification\b',
            'oracle': r'\boracle\s*certified\b|\boca\b|\bocp\b',
            'cisco': r'\bcisco\s*certified\b|\bccna\b|\bccnp\b',
            'comptia': r'\bcomptia\b|\ba\+\b|\bnetwork\+\b|\bsecurity\+\b',
            'pmp': r'\bpmp\b|\bproject\s*management\s*professional\b',
            'itil': r'\bitil\b|\bitil\s*foundation\b',
            'scrum': r'\bscrum\s*master\b|\bpsm\b|\bcsm\b'
        }
        
        for cert, pattern in cert_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                if cert not in entities['CERTIFICATION']:
                    entities['CERTIFICATION'].append(cert)
        
        # Ensure all entity types exist in the result
        for entity_type in ['AGE', 'GENDER', 'ADDRESS', 'SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION']:
            if entity_type not in entities:
                entities[entity_type] = []
                
        return dict(entities)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using NER with regex fallback"""
        if self.ner_pipeline:
            return self.extract_entities_with_ner(text)
        return self.extract_entities_with_regex(text)

    def analyze_resume(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        """Analyze resume against job requirements"""
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
                
                # Calculate matches and scores using enhanced similarity for relevant entities
                if entity_type in ['SKILLS', 'EDUCATION', 'CERTIFICATION']:
                    resume_set = set(resume_entities[entity_type])
                    job_set = set(job_entities[entity_type])
                    
                    # Use enhanced similarity calculation
                    score = self.calculate_enhanced_similarity(resume_set, job_set)
                    
                    # Find matching and missing items
                    matches = set()
                    for item1 in resume_set:
                        for item2 in job_set:
                            if (self.normalize_text(item1) == self.normalize_text(item2) or
                                self.skill_mapping.get(self.normalize_text(item1)) == 
                                self.skill_mapping.get(self.normalize_text(item2))):
                                matches.add(item1)
                                break
                    
                    missing = {item for item in job_set if not any(
                        self.skill_mapping.get(self.normalize_text(item)) == 
                        self.skill_mapping.get(self.normalize_text(x))
                        for x in resume_set
                    )}
                    
                    entity_analysis[entity_type]['matching'] = list(matches)
                    entity_analysis[entity_type]['missing'] = list(missing)
                    entity_analysis[entity_type]['matching_score'] = score * 100
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
            
            return {
                'entity_analysis': entity_analysis,
                'overall_match': overall_match,
                'score_breakdown': score_breakdown,
                'suitability_status': suitability_status
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

    def _extract_years_experience(self, experience_list: List[str]) -> float:
        """Extract years of experience from experience descriptions"""
        years = 0.0
        for exp in experience_list:
            # Look for patterns like "X years" or "X+ years"
            matches = re.findall(r'(\d+)(?:\+)?\s*years?', exp.lower())
            if matches:
                years = max(years, float(matches[0]))
        return years

