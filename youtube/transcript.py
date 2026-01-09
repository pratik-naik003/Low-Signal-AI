from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from urllib.parse import urlparse, parse_qs


def get_youtube_video_id(url: str) -> str | None:
    parsed_url = urlparse(url)

    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        query = parse_qs(parsed_url.query)
        if "v" in query:
            return query["v"][0]

        path_parts = parsed_url.path.split("/")
        if "embed" in path_parts or "shorts" in path_parts:
            return path_parts[-1]

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    return None


def get_youtube_transcript(url: str) -> str:
    video_id = get_youtube_video_id(url)

    if not video_id:
        raise ValueError("Invalid YouTube URL")

    try:
        # ✅ SAFE & RELIABLE FLOW
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        transcript = transcript_list.find_transcript(['en']).fetch()

        return " ".join(item.text for item in transcript)

    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video")

    except Exception as e:
        raise ValueError(f"Error fetching transcript: {e}")


