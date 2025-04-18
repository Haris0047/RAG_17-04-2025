from __future__ import annotations

import numpy as np
import sec_parser as sp

# SEC Parser
from sec_parser import TextElement, TitleElement, TopSectionTitle
from sklearn.metrics.pairwise import cosine_similarity

# Convert raw documents' format


# import helper


def combine_sentences(sentences, buffer_size=1):
    # Go through each sentence dict
    for i in range(len(sentences)):

        # Create a string that will hold the sentences which are joined
        combined_sentence = ""

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]["sentence"] + " "

        # Add the current sentence
        combined_sentence += sentences[i]["sentence"]

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += " " + sentences[j]["sentence"]

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]["combined_sentence"] = combined_sentence

    return sentences


def processing_html2txt(html):
    elements = sp.Edgar10QParser().parse(html)
    top_level_sections = [
        item for part in sp.TreeBuilder().build(elements) for item in part.children
    ]

    # Get the levels we need in the document
    levels = sorted(
        {
            k.semantic_element.level
            for k in top_level_sections
            if isinstance(k.semantic_element, (sp.TopSectionTitle, sp.TitleElement))
        }
    )
    level_to_markdown = {level: "#" * (i + 2) for i, level in enumerate(levels)}

    # Function to extract all the text (excluding tables), and cnvert to markdown format
    def convert_to_markdown(sections):
        markdown = ""
        for section in sections:
            if isinstance(section.semantic_element, (TopSectionTitle, TitleElement)):
                markdown += f"{level_to_markdown.get(section.semantic_element.level, '#')} {section.semantic_element.text}\n"
            elif isinstance(section.semantic_element, TextElement):
                markdown += f"{section.semantic_element.text}\n"
            for child in section.get_descendants():
                if isinstance(child.semantic_element, (TopSectionTitle, TitleElement)):
                    markdown += f"{level_to_markdown.get(child.semantic_element.level, '#')} {child.semantic_element.text}\n"
                elif isinstance(child.semantic_element, TextElement):
                    markdown += f"{child.semantic_element.text}\n"
        return markdown

    raw_essay = convert_to_markdown(top_level_sections)
    return raw_essay


def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]["combined_sentence_embedding"]
        embedding_next = sentences[i + 1]["combined_sentence_embedding"]

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]["distance_to_next"] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences


def calculate_chunk_sizes(sentences, distances, threshold):
    # Determine the distance threshold
    breakpoint_distance_threshold = np.percentile(distances, threshold)

    # Find the indices of distances above the threshold
    indices_above_thresh = [
        i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
    ]

    # Initialize the start index
    start_index = 0
    chunks = []

    # Iterate through the breakpoints to slice the sentences
    for index in indices_above_thresh:
        end_index = index
        group = sentences[start_index : end_index + 1]
        combined_text = " ".join([d["sentence"] for d in group])
        chunks.append(combined_text)
        start_index = index + 1

    # The last group, if any sentences remain
    if start_index < len(sentences):
        combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
        chunks.append(combined_text)

    return chunks


# Function to find appropriate threshold
def find_appropriate_threshold(sentences, distances, initial_threshold, ceiling):
    threshold = initial_threshold
    while threshold > 0:
        chunks = calculate_chunk_sizes(sentences, distances, threshold)
        chunk_sizes = [len(chunk.split()) for chunk in chunks]
        if max(chunk_sizes) <= ceiling:
            break
        threshold -= 1
    return threshold, chunks, chunk_sizes
