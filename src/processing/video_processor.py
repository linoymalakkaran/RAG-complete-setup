"""
Video Processing Module with Whisper Integration

This module provides functionality to process video files by:
1. Extracting audio from video files
2. Transcribing audio using OpenAI Whisper
3. Creating searchable chunks with timestamps
4. Integrating with the RAG system for video content search
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

# Try to import video processing libraries
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not installed. Run: pip install openai-whisper")

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("MoviePy not installed. Run: pip install moviepy")


@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed video content."""
    text: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    @property
    def start_timestamp(self) -> str:
        return self.format_timestamp(self.start_time)
    
    @property
    def end_timestamp(self) -> str:
        return self.format_timestamp(self.end_time)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class VideoDocument:
    """Represents a processed video with its transcript."""
    video_path: str
    title: str
    transcript_segments: List[TranscriptSegment]
    metadata: Dict[str, Any]
    full_transcript: str
    language: str = "en"
    
    def get_segment_by_time(self, timestamp: float) -> Optional[TranscriptSegment]:
        """Get the transcript segment at a specific timestamp."""
        for segment in self.transcript_segments:
            if segment.start_time <= timestamp <= segment.end_time:
                return segment
        return None
    
    def search_text(self, query: str) -> List[TranscriptSegment]:
        """Search for text in transcript segments."""
        query_lower = query.lower()
        matching_segments = []
        for segment in self.transcript_segments:
            if query_lower in segment.text.lower():
                matching_segments.append(segment)
        return matching_segments


class VideoProcessor:
    """
    Process video files for RAG integration.
    
    Features:
    - Extract audio from video files
    - Transcribe using Whisper (multiple model sizes)
    - Create timestamped segments
    - Generate metadata
    """
    
    def __init__(
        self,
        whisper_model: str = "base",
        chunk_duration: int = 30,
        device: str = "cpu",
        language: Optional[str] = None
    ):
        """
        Initialize the video processor.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            chunk_duration: Duration of each transcript chunk in seconds
            device: Device to run Whisper on (cpu, cuda)
            language: Language code (None for auto-detection)
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper not installed. Run: pip install openai-whisper")
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy not installed. Run: pip install moviepy")
        
        self.whisper_model_name = whisper_model
        self.chunk_duration = chunk_duration
        self.device = device
        self.language = language
        
        logger.info(f"Loading Whisper model: {whisper_model}")
        self.model = whisper.load_model(whisper_model, device=device)
        logger.info("Whisper model loaded successfully")
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            output_path: Path to save audio file (optional)
            
        Returns:
            Path to extracted audio file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create temporary audio file if no output path specified
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            video_name = Path(video_path).stem
            output_path = os.path.join(temp_dir, f"{video_name}_audio.wav")
        
        logger.info(f"Extracting audio from: {video_path}")
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(
                output_path,
                logger=None,  # Suppress moviepy logs
                verbose=False
            )
            video.close()
            logger.info(f"Audio extracted to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise
    
    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Language code (overrides default)
            
        Returns:
            Whisper transcription result
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing audio: {audio_path}")
        
        # Prepare options
        options = {}
        if language or self.language:
            options["language"] = language or self.language
        
        try:
            result = self.model.transcribe(audio_path, **options)
            logger.info(f"Transcription complete. Language: {result.get('language', 'unknown')}")
            return result
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    
    def create_segments(
        self,
        transcription: Dict[str, Any],
        chunk_duration: Optional[int] = None
    ) -> List[TranscriptSegment]:
        """
        Create transcript segments from Whisper output.
        
        Args:
            transcription: Whisper transcription result
            chunk_duration: Override default chunk duration
            
        Returns:
            List of transcript segments
        """
        chunk_dur = chunk_duration or self.chunk_duration
        segments = []
        
        # Use Whisper's built-in segments
        whisper_segments = transcription.get("segments", [])
        
        if not whisper_segments:
            # Fallback: create single segment from full text
            segments.append(TranscriptSegment(
                text=transcription.get("text", ""),
                start_time=0,
                end_time=0,
                confidence=None
            ))
            return segments
        
        # Group segments by chunk duration
        current_chunk_text = []
        current_chunk_start = whisper_segments[0]["start"]
        current_chunk_end = current_chunk_start
        
        for seg in whisper_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_text = seg["text"].strip()
            
            # Check if adding this segment exceeds chunk duration
            if seg_end - current_chunk_start > chunk_dur and current_chunk_text:
                # Save current chunk
                segments.append(TranscriptSegment(
                    text=" ".join(current_chunk_text),
                    start_time=current_chunk_start,
                    end_time=current_chunk_end,
                    confidence=None
                ))
                
                # Start new chunk
                current_chunk_text = [seg_text]
                current_chunk_start = seg_start
                current_chunk_end = seg_end
            else:
                # Add to current chunk
                current_chunk_text.append(seg_text)
                current_chunk_end = seg_end
        
        # Add final chunk
        if current_chunk_text:
            segments.append(TranscriptSegment(
                text=" ".join(current_chunk_text),
                start_time=current_chunk_start,
                end_time=current_chunk_end,
                confidence=None
            ))
        
        logger.info(f"Created {len(segments)} transcript segments")
        return segments
    
    def process_video(
        self,
        video_path: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        keep_audio: bool = False
    ) -> VideoDocument:
        """
        Process a video file end-to-end.
        
        Args:
            video_path: Path to video file
            title: Video title (uses filename if not provided)
            metadata: Additional metadata
            keep_audio: Whether to keep extracted audio file
            
        Returns:
            VideoDocument with transcript and metadata
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Use filename as title if not provided
        if title is None:
            title = Path(video_path).stem
        
        logger.info(f"Processing video: {title}")
        
        # Step 1: Extract audio
        audio_path = self.extract_audio(video_path)
        
        try:
            # Step 2: Transcribe
            transcription = self.transcribe_audio(audio_path)
            
            # Step 3: Create segments
            segments = self.create_segments(transcription)
            
            # Step 4: Build metadata
            video_metadata = metadata or {}
            video_metadata.update({
                "video_path": video_path,
                "audio_path": audio_path if keep_audio else None,
                "duration": segments[-1].end_time if segments else 0,
                "num_segments": len(segments),
                "language": transcription.get("language", "unknown"),
                "whisper_model": self.whisper_model_name
            })
            
            # Step 5: Create VideoDocument
            video_doc = VideoDocument(
                video_path=video_path,
                title=title,
                transcript_segments=segments,
                metadata=video_metadata,
                full_transcript=transcription.get("text", ""),
                language=transcription.get("language", "unknown")
            )
            
            logger.info(f"Video processing complete: {title}")
            return video_doc
            
        finally:
            # Clean up audio file unless keep_audio is True
            if not keep_audio and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.info("Temporary audio file removed")
                except Exception as e:
                    logger.warning(f"Could not remove audio file: {e}")
    
    def process_video_directory(
        self,
        directory: str,
        extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
        **kwargs
    ) -> List[VideoDocument]:
        """
        Process all video files in a directory.
        
        Args:
            directory: Path to directory containing videos
            extensions: Video file extensions to process
            **kwargs: Additional arguments passed to process_video
            
        Returns:
            List of VideoDocument objects
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        video_files = []
        for ext in extensions:
            video_files.extend(Path(directory).glob(f"*{ext}"))
        
        logger.info(f"Found {len(video_files)} video files in {directory}")
        
        video_documents = []
        for video_file in video_files:
            try:
                video_doc = self.process_video(str(video_file), **kwargs)
                video_documents.append(video_doc)
            except Exception as e:
                logger.error(f"Error processing {video_file}: {e}")
        
        logger.info(f"Successfully processed {len(video_documents)} videos")
        return video_documents


class VideoRAGIntegration:
    """
    Integrate video processing with RAG system.
    
    Converts VideoDocument objects to searchable chunks for vector store.
    """
    
    @staticmethod
    def video_to_documents(
        video_doc: VideoDocument,
        include_timestamps: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Convert VideoDocument to RAG-compatible document format.
        
        Args:
            video_doc: VideoDocument to convert
            include_timestamps: Include timestamp info in text
            
        Returns:
            List of document dictionaries for vector store
        """
        documents = []
        
        for i, segment in enumerate(video_doc.transcript_segments):
            # Build document text
            text = segment.text
            if include_timestamps:
                text = f"[{segment.start_timestamp} - {segment.end_timestamp}] {text}"
            
            # Build metadata
            doc_metadata = {
                "source": video_doc.video_path,
                "title": video_doc.title,
                "type": "video_transcript",
                "segment_index": i,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "start_timestamp": segment.start_timestamp,
                "end_timestamp": segment.end_timestamp,
                "duration": segment.duration,
                "language": video_doc.language,
                **video_doc.metadata
            }
            
            documents.append({
                "text": text,
                "metadata": doc_metadata
            })
        
        return documents
    
    @staticmethod
    def format_video_context(
        retrieved_docs: List[Dict[str, Any]],
        max_segments: int = 5
    ) -> str:
        """
        Format retrieved video segments for context.
        
        Args:
            retrieved_docs: Retrieved documents from vector store
            max_segments: Maximum number of segments to include
            
        Returns:
            Formatted context string
        """
        video_segments = [
            doc for doc in retrieved_docs
            if doc.get("metadata", {}).get("type") == "video_transcript"
        ]
        
        if not video_segments:
            return ""
        
        # Group by video
        videos = {}
        for doc in video_segments[:max_segments]:
            metadata = doc.get("metadata", {})
            title = metadata.get("title", "Unknown Video")
            
            if title not in videos:
                videos[title] = []
            
            videos[title].append({
                "text": doc.get("text", ""),
                "timestamp": metadata.get("start_timestamp", "00:00:00"),
                "segment_index": metadata.get("segment_index", 0)
            })
        
        # Format output
        context_parts = ["# Video Transcript Excerpts\n"]
        
        for title, segments in videos.items():
            context_parts.append(f"\n## {title}\n")
            
            # Sort by segment index
            segments.sort(key=lambda x: x["segment_index"])
            
            for seg in segments:
                context_parts.append(f"**{seg['timestamp']}**: {seg['text']}\n")
        
        return "\n".join(context_parts)


# Convenience function
def process_video_for_rag(
    video_path: str,
    whisper_model: str = "base",
    chunk_duration: int = 30,
    **kwargs
) -> Tuple[VideoDocument, List[Dict[str, Any]]]:
    """
    Convenience function to process video and get RAG-ready documents.
    
    Args:
        video_path: Path to video file
        whisper_model: Whisper model size
        chunk_duration: Chunk duration in seconds
        **kwargs: Additional arguments for VideoProcessor.process_video
        
    Returns:
        Tuple of (VideoDocument, list of RAG documents)
    """
    processor = VideoProcessor(
        whisper_model=whisper_model,
        chunk_duration=chunk_duration
    )
    
    video_doc = processor.process_video(video_path, **kwargs)
    rag_docs = VideoRAGIntegration.video_to_documents(video_doc)
    
    return video_doc, rag_docs
