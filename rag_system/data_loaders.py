import os
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def extract_youtube_video_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)",
        r"youtube\.com\/watch\?.*v=([^&\n?#]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_youtube_transcript(video_id):
    """Get YouTube transcript using multiple methods"""
    transcript_text = ""
    video_title = f"YouTube Video: {video_id}"

    # Method 1: Try youtube-transcript-api with correct syntax
    try:
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            TranscriptsDisabled,
            NoTranscriptFound,
        )

        # Get transcript directly - try different language codes
        language_codes = ["en", "en-US", "en-GB", "en-IN"]

        for lang_code in language_codes:
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=[lang_code]
                )
                break
            except:
                continue
        else:
            # If no English transcript, get any available and try to translate
            try:
                # Get available transcripts
                available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript_list = list(available_transcripts)
                if transcript_list:
                    # Try to get the first available transcript
                    first_transcript = transcript_list[0]
                    if hasattr(first_transcript, "fetch"):
                        transcript_data = first_transcript.fetch()
                    else:
                        transcript_data = YouTubeTranscriptApi.get_transcript(
                            video_id, languages=[first_transcript.language_code]
                        )
                else:
                    transcript_data = []
            except:
                transcript_data = []

        if transcript_data:
            # Combine transcript entries with timestamps
            transcript_parts = []
            for entry in transcript_data:
                timestamp = entry.get("start", 0)
                text = entry.get("text", "").strip()
                if text:
                    # Clean up text
                    text = re.sub(
                        r"\[.*?\]", "", text
                    )  # Remove [Music], [Applause] etc.
                    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
                    text = text.replace("\n", " ").strip()
                    if text:
                        # Format: [MM:SS] Text
                        minutes = int(timestamp // 60)
                        seconds = int(timestamp % 60)
                        transcript_parts.append(f"[{minutes:02d}:{seconds:02d}] {text}")

            if transcript_parts:
                transcript_text = "\n".join(transcript_parts)
                print(
                    f"✅ Successfully extracted transcript using youtube-transcript-api"
                )
                print(f"   Found {len(transcript_parts)} transcript segments")
                return transcript_text, video_title

    except Exception as e:
        print(f"youtube-transcript-api method failed: {e}")

    # Method 2: Try yt-dlp (more reliable than pytube)
    try:
        import subprocess
        import json
        import tempfile
        import os

        # Use yt-dlp to extract info and subtitles
        cmd = [
            "yt-dlp",
            "--write-auto-sub",  # Write automatic subtitles
            "--write-sub",  # Write subtitles
            "--sub-lang",
            "en",  # English subtitles
            "--skip-download",  # Don't download video
            "--print",
            "title",  # Print title
            "--print",
            "duration",  # Print duration
            f"https://www.youtube.com/watch?v={video_id}",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                cmd, cwd=temp_dir, capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                # Look for subtitle files
                for file in os.listdir(temp_dir):
                    if file.endswith(".vtt") or file.endswith(".srt"):
                        with open(
                            os.path.join(temp_dir, file), "r", encoding="utf-8"
                        ) as f:
                            subtitle_content = f.read()

                        # Parse VTT/SRT content
                        lines = subtitle_content.split("\n")
                        clean_lines = []

                        for line in lines:
                            line = line.strip()
                            # Skip timing lines, numbers, and WEBVTT headers
                            if (
                                line
                                and not line.startswith("WEBVTT")
                                and not line.isdigit()
                                and not "-->" in line
                                and not line.startswith("NOTE")
                            ):

                                # Clean up the text
                                line = re.sub(r"<[^>]*>", "", line)  # Remove HTML tags
                                line = re.sub(
                                    r"\[.*?\]", "", line
                                )  # Remove [Music] etc.
                                if line.strip():
                                    clean_lines.append(line.strip())

                        if clean_lines:
                            transcript_text = "\n".join(clean_lines)
                            # Get video title from output
                            output_lines = result.stdout.strip().split("\n")
                            if output_lines:
                                video_title = output_lines[0]

                            print(f"✅ Successfully extracted transcript using yt-dlp")
                            return transcript_text, video_title

    except Exception as e:
        print(f"yt-dlp method failed: {e}")

    # Method 3: Try requests + BeautifulSoup to scrape transcript
    try:
        import requests
        from bs4 import BeautifulSoup
        import json

        # Get the YouTube page
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")

            # Try to find video title
            title_element = soup.find("meta", property="og:title")
            if title_element:
                video_title = title_element.get("content", video_title)

            # Look for transcript data in script tags
            scripts = soup.find_all("script")
            for script in scripts:
                if script.string and "captionTracks" in script.string:
                    # This is a simplified approach - in practice, you'd need to parse the JS
                    print("Found potential transcript data in page source")
                    break

    except Exception as e:
        print(f"Web scraping method failed: {e}")

    return transcript_text, video_title


def load_youtube_content(url):
    """Load YouTube video content with robust transcript extraction"""
    video_id = extract_youtube_video_id(url)
    if not video_id:
        print(f"❌ Could not extract video ID from: {url}")
        return []

    print(f"🔄 Processing YouTube video: {video_id}")

    # Get transcript
    transcript_text, video_title = get_youtube_transcript(video_id)

    if not transcript_text or len(transcript_text.strip()) < 50:
        print(f"❌ No transcript available or transcript too short for: {url}")
        print(f"   This video may not have captions/subtitles enabled.")
        return []

    # Create document with transcript
    doc = Document(
        page_content=transcript_text,
        metadata={
            "source": url,
            "title": video_title,
            "video_id": video_id,
            "type": "youtube_transcript",
            "length": len(transcript_text),
        },
    )

    print(f"✅ Successfully loaded YouTube transcript:")
    print(f"   Title: {video_title}")
    print(f"   Length: {len(transcript_text)} characters")
    print(f"   Preview: {transcript_text[:200]}...")

    return [doc]


def test_youtube_url(url):
    """Test function to check if YouTube URL works"""
    print(f"\n🧪 Testing YouTube URL: {url}")
    docs = load_youtube_content(url)
    if docs:
        print(f"✅ Test successful! Loaded {len(docs)} document(s)")
        return True
    else:
        print(f"❌ Test failed!")
        return False


def load_and_process_data(uploaded_files, urls_input):
    """Main function to load and process all data sources"""
    documents = []

    # Process uploaded files
    if uploaded_files:
        print("\n📁 Processing uploaded files...")
        for uploaded_file in uploaded_files:
            try:
                # Save the file temporarily
                file_path = Path(uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load the document using the appropriate loader
                if file_path.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                else:
                    from langchain_community.document_loaders import TextLoader

                    loader = TextLoader(str(file_path), encoding="utf-8")

                docs = loader.load()
                documents.extend(docs)
                print(f"✅ Successfully loaded file: {uploaded_file.name}")

            except Exception as e:
                print(f"❌ Failed to load file {uploaded_file.name}: {e}")
            finally:
                if file_path.exists():
                    os.remove(file_path)

    # Process URLs
    if urls_input:
        print("\n🌐 Processing URLs...")
        urls = [url.strip() for url in urls_input.split("\n") if url.strip()]

        for url in urls:
            try:
                if "youtube.com" in url or "youtu.be" in url:
                    youtube_docs = load_youtube_content(url)
                    documents.extend(youtube_docs)
                else:
                    print(f"🔄 Loading web page: {url}")
                    loader = WebBaseLoader(
                        url,
                        header_template={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        },
                    )
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"✅ Successfully loaded URL: {url}")

            except Exception as e:
                print(f"❌ Failed to load {url}: {e}")

    if not documents:
        print("\n⚠️ No documents were successfully loaded.")
        return []

    # Split documents into chunks
    print(f"\n📄 Splitting {len(documents)} document(s) into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    try:
        docs = text_splitter.split_documents(documents)
        print(f"✅ Successfully split into {len(docs)} chunks.")

        # Print summary of loaded content
        print(f"\n📊 Content Summary:")
        for doc in documents:
            doc_type = doc.metadata.get("type", "unknown")
            title = doc.metadata.get("title", "Unknown Title")
            length = len(doc.page_content)
            print(f"   - {doc_type}: {title} ({length} chars)")

        return docs

    except Exception as e:
        print(f"❌ Error splitting documents: {e}")
        return documents