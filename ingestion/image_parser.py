# ingestion/image_parser.py
import anthropic
from typing import List
from ingestion.pdf_parser import ParsedElement

class ImageDescriber:
    """
    Uses Claude Haiku vision to describe embedded images.
    Cost: ~$0.003 per image. Falls back gracefully on failure.
    """
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def describe_batch(self, elements: List[ParsedElement]) -> List[ParsedElement]:
        image_elements = [
            e for e in elements
            if e.content_type == "image" and e.raw_image_b64
        ]

        for elem in image_elements:
            try:
                response = self.client.messages.create(
                    model="claude-haiku-3-5-20241022",
                    max_tokens=300,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": elem.raw_image_b64
                                }
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Describe this image concisely for a RAG system. "
                                    "If it contains text, extract it verbatim. "
                                    "If it's a chart/graph, describe axes, trends, and key values. "
                                    "If it's a diagram, describe the flow and relationships. "
                                    "Max 200 words."
                                )
                            }
                        ]
                    }]
                )
                elem.content = response.content[0].text
                elem.raw_image_b64 = None  # Release memory allocation
            except Exception as e:
                print(f"[WARN] Image description failed: {e}")
                elem.content = "[Image: description unavailable]"

        return elements
