import os
import json
import base64
from google import genai
from google.genai import types
from google.oauth2 import service_account
from typing import Dict, List, Any
from dotenv import load_dotenv  # Add this import

class VideoContentRAG:
    def __init__(self, credentials_path: str, gemini_api_key: str = None):
        """
        Initialize the Video Content RAG system with GCP credentials.
        
        Args:
            credentials_path: Path to the service account JSON file
            gemini_api_key: Optional Gemini API key (if not provided, will look for GEMINI_API_KEY env var)
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Set up authentication
        self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
        
        # Initialize Vertex AI with the project details from credentials
        with open(credentials_path) as f:
            creds_data = json.load(f)
            self.project_id = creds_data["project_id"]
        
        # Initialize Gemini client - try multiple ways to get the API key
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not self.gemini_api_key:
            print("Warning: No Gemini API key found. Please provide it directly or set the GEMINI_API_KEY environment variable.")
            # Instead of raising an error, we'll prompt for the key
            self.gemini_api_key = input("Please enter your Gemini API key: ")
            if not self.gemini_api_key:
                raise ValueError("Gemini API key must be provided to continue")
        
        # Initialize the Gemini client
        self.genai_client = genai.Client(api_key=self.gemini_api_key)
        self.model_name = "gemini-2.0-flash-thinking-exp-01-21"
        
        # Store for video analysis results
        self.video_data = {}
        
    def load_video_analysis(self, analysis_json: str, video_id: str = None):
        """
        Load and process video analysis data.
        
        Args:
            analysis_json: Path to JSON file or JSON string with analysis results
            video_id: Optional identifier for the video
        """
        # Determine if input is a file path or JSON string
        if os.path.exists(analysis_json):
            with open(analysis_json, 'r') as f:
                data = json.load(f)
        else:
            try:
                data = json.loads(analysis_json)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON input")
        
        # Extract video ID from the data if not provided
        if not video_id and 'annotation_results' in data:
            first_result = data['annotation_results'][0]
            if 'input_uri' in first_result:
                video_id = first_result['input_uri'].split('/')[-1]
            else:
                video_id = f"video_{len(self.video_data) + 1}"
        
        # Process and structure the data for RAG
        processed_data = self._process_video_data(data)
        self.video_data[video_id] = processed_data
        
        return f"Loaded analysis for video: {video_id}"
    
    def _process_video_data(self, data: Dict) -> Dict:
        """
        Process and structure the video analysis data for efficient retrieval.
        
        Args:
            data: Raw video analysis JSON data
            
        Returns:
            Processed and structured data
        """
        processed = {
            "segment_labels": {},
            "shot_labels": {},
            "objects": {},
            "persons": [],
            "texts": [],
            "speech": [],
            "summary": {
                "main_entities": set(),
                "duration": None,
                "scene_count": 0
            }
        }
        
        if 'annotation_results' not in data:
            return processed
        
        for result in data['annotation_results']:
            # Process segment labels (whole video level)
            if 'segment_label_annotations' in result:
                for annotation in result['segment_label_annotations']:
                    entity = annotation['entity']['description']
                    confidence = max([seg['confidence'] for seg in annotation['segments']])
                    processed['segment_labels'][entity] = confidence
                    processed['summary']['main_entities'].add(entity)
            
            # Process shot labels (scene level)
            if 'shot_label_annotations' in result:
                for annotation in result['shot_label_annotations']:
                    entity = annotation['entity']['description']
                    if entity not in processed['shot_labels']:
                        processed['shot_labels'][entity] = []
                    
                    for segment in annotation['segments']:
                        start = self._time_offset_to_seconds(segment['segment']['start_time_offset'])
                        end = self._time_offset_to_seconds(segment['segment']['end_time_offset'])
                        processed['shot_labels'][entity].append({
                            'confidence': segment['confidence'],
                            'start_time': start,
                            'end_time': end,
                            'duration': end - start
                        })
                        processed['summary']['scene_count'] = max(
                            processed['summary']['scene_count'], 
                            len(processed['shot_labels'][entity])
                        )
            
            # Get the video duration from the segment information if available
            if 'segment' in result:
                processed['summary']['duration'] = self._time_offset_to_seconds(
                    result['segment']['end_time_offset']
                )
        
        # Convert set to list for JSON serialization
        processed['summary']['main_entities'] = list(processed['summary']['main_entities'])
        
        return processed
    
    def _time_offset_to_seconds(self, time_offset: Dict) -> float:
        """
        Convert a time offset object to seconds.
        
        Args:
            time_offset: Dictionary containing seconds and nanos
            
        Returns:
            Time in seconds as a float
        """
        if not time_offset:
            return 0.0
        
        seconds = time_offset.get('seconds', 0)
        nanos = time_offset.get('nanos', 0)
        
        return float(seconds) + (float(nanos) / 1_000_000_000)
    
    def _format_context_for_llm(self, video_data: Dict) -> str:
        """
        Format the video data as context for the LLM with clean, readable formatting.
        
        Args:
            video_data: Processed video data
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for vid_id, data in video_data.items():
            context_parts.append(f"Video ID: {vid_id}")
            
            # Add duration
            if data['summary']['duration']:
                duration_mins = data['summary']['duration'] / 60
                context_parts.append(f"Duration: {duration_mins:.2f} minutes")
            
            # Add main entities/concepts
            if data['summary']['main_entities']:
                context_parts.append(f"Main concepts detected: {', '.join(data['summary']['main_entities'])}")
            
            # Add segment labels (whole video concepts)
            if data['segment_labels']:
                context_parts.append("Video-level labels:")
                for label, confidence in sorted(data['segment_labels'].items(), key=lambda x: x[1], reverse=True):
                    context_parts.append(f"- {label} (confidence: {confidence:.2f})")
            
            # Add shot labels (scene-specific concepts)
            if data['shot_labels']:
                context_parts.append("Scene-specific labels:")
                for label, scenes in data['shot_labels'].items():
                    context_parts.append(f"- {label} appears in {len(scenes)} scene(s):")
                    for i, scene in enumerate(scenes):
                        start_min = scene['start_time'] / 60
                        end_min = scene['end_time'] / 60
                        context_parts.append(f"  * Scene {i+1}: {start_min:.2f}-{end_min:.2f} min (confidence: {scene['confidence']:.2f})")
            
            # Count persons if available
            if data['persons']:
                person_count = len(data['persons'])
                context_parts.append(f"Number of people detected: {person_count}")
        
        return "\n".join(context_parts)
    
    def answer_query(self, query: str, video_id: str = None) -> str:
        """
        Answer a query about the video content using Gemini.
        
        Args:
            query: User's question about the video
            video_id: Optional video ID to query (if None, use all videos)
            
        Returns:
            Answer to the user's query
        """
        # Determine which video data to use
        if video_id and video_id in self.video_data:
            context_data = {video_id: self.video_data[video_id]}
        else:
            context_data = self.video_data
        
        if not context_data:
            return "No video analysis data loaded. Please load video analysis first."
        
        # Format context information for the LLM
        context = self._format_context_for_llm(context_data)
        
        # Prepare prompt for Gemini with explicit formatting instructions
        prompt = f"""
        Given the following video analysis data:
        
        {context}
        
        Please answer this question about the video content:
        {query}
        
        Important formatting instructions:
        - Do not include any HTML, CSS, markdown formatting, or styling tags in your response
        - Format your answer in simple plain text with standard bullet points (using * or - characters)
        - Use simple time formats like "1:30" for timestamps, not styled versions
        - For percentages, use simple formats like "41%" without any special styling
        - Keep the response focused and concise
        - Use only plain text formatting that would work in a basic text editor
        - give answer in bullet point
        
        Base your answer solely on the information provided in the video analysis data. 
        If the information is not available in the data, please indicate that.
        """
        
        # Call Gemini model
        response = self._call_gemini_model(prompt)
        return response
    
    def _call_gemini_model(self, prompt: str) -> str:
        """
        Call Gemini model for text generation.
        
        Args:
            prompt: The formatted prompt
            
        Returns:
            Generated response
        """
        try:
            # Create content for Gemini
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            # Configure generation parameters
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",  # Force plain text output
                temperature=0.2,
                top_p=0.8,
                top_k=40,
                max_output_tokens=1024,
            )
            
            # Generate response
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )
            
            return response.text
        except Exception as e:
            return f"Error calling Gemini model: {str(e)}"
    
    def process_video(self, video_path: str, features: List[str] = None):
        """
        Process a new video using Video Intelligence API.
        
        Args:
            video_path: GCS URI to the video file
            features: List of features to detect (e.g., ['LABEL_DETECTION', 'OBJECT_TRACKING'])
            
        Returns:
            Operation ID for the long-running operation
        """
        # This would be implemented using the Video Intelligence API
        # For now, we'll return a placeholder message
        return "Video processing with Video Intelligence API would be implemented here."

# Example usage
def example_usage():
    # Initialize the VideoContentRAG system
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # If you want to hardcode the API key for testing, uncomment and replace with your key
    # api_key = "your-api-key-here"
    
    try:
        video_rag = VideoContentRAG('chat bot.json', gemini_api_key=api_key)
        
        # Load sample video analysis
        sample_json = """
        {
          "annotation_results": [ {
            "input_uri": "/test-duck/company-vids/test_5-generation-bakers.mp4",
            "segment": {
              "start_time_offset": {},
              "end_time_offset": {
                "seconds": 252,
                "nanos": 669083000
              }
            },
            "segment_label_annotations": [ {
              "entity": {
                "entity_id": "/m/018y_6",
                "description": "inventory",
                "language_code": "en-US"
              },
              "segments": [ {
                "segment": {
                  "start_time_offset": {},
                  "end_time_offset": {
                    "seconds": 252,
                    "nanos": 669083000
                  }
                },
                "confidence": 0.3776648
              } ]
            }, {
              "entity": {
                "entity_id": "/m/02wbm",
                "description": "food",
                "language_code": "en-US"
              },
              "segments": [ {
                "segment": {
                  "start_time_offset": {},
                  "end_time_offset": {
                    "seconds": 252,
                    "nanos": 669083000
                  }
                },
                "confidence": 0.58392262
              } ]
            } ],
            "shot_label_annotations": [ {
              "entity": {
                "entity_id": "/m/01c8br",
                "description": "street",
                "language_code": "en-US"
              },
              "segments": [ {
                "segment": {
                  "start_time_offset": {"seconds": 77, "nanos": 368958000},
                  "end_time_offset": {"seconds": 80, "nanos": 288541000}
                },
                "confidence": 0.73337024
              } ]
            } ]
          } ]
        }
        """
        video_rag.load_video_analysis(sample_json)
        
        # Answer a query
        answer = video_rag.answer_query("What is the main content of the video?")
        print(answer)
        
        answer = video_rag.answer_query("How long is the video?")
        print(answer)
        
        answer = video_rag.answer_query("Are there any streets shown in the video?")
        print(answer)
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install python-dotenv google-generativeai")

if __name__ == "__main__":
    example_usage()