import subprocess
import re
import os
import random

from decouple import config

SG_PATH = config("SG_PATH")

LLAMA_CLI_PATH = f"{SG_PATH}/llama.cpp/build/bin/llama-cli"
MODEL_PATH = f"{SG_PATH}/llama.cpp/hf_models/LLaMA-3.2-3B-Instruct-Q4_K_M.gguf"
OUTPUT_FILE = f"{SG_PATH}/selected_story.txt"

def parse_paragraph_outputs(file_path):
    """
    Parses the file with outlines and paragraph outputs.
    Returns a list of tuples (story_num, paragraph_text).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    paragraphs = []
    story_num = None

    for line in lines:
        line_strip = line.strip()
        # Detect outline header line for story number
        m_outline = re.match(r"--- Story Outline #(\d+) ---", line_strip)
        if m_outline:
            story_num = int(m_outline.group(1))
            continue
        
        # Detect paragraph output header line with event
        m_paragraph = re.match(r"--- Paragraph Output for Event: '.+' ---", line_strip)
        if m_paragraph and story_num is not None:
            # Next line(s) will be paragraph text until empty line or next header
            paragraph_lines = []
            idx = lines.index(line) + 1
            while idx < len(lines) and not lines[idx].strip().startswith("---"):
                paragraph_lines.append(lines[idx].rstrip('\n'))
                idx += 1
            
            paragraph_text = " ".join(paragraph_lines).strip()
            if paragraph_text:
                paragraphs.append((story_num, paragraph_text))

    return paragraphs

def load_stories(filename):
    if not os.path.exists(filename):
        print(f"Story file '{filename}' not found.")
        return []
    with open(filename, "r", encoding="utf-8") as f:
        stories = [line.strip() for line in f if line.strip()]
    return stories

def ask_compatibility_score(initial_paragraph, story_text):
    """
    Sends prompt to llama-cli asking for compatibility score (0 to 1)
    between initial_paragraph and story_text.
    Expects the model to output only a decimal number in the output.
    """
    prompt = f"""
    
    I want to match a single paragraph to a full children's story that could be applied to the paragraph's context.
    
    Compare the following two passages:
    Passage 1 (Initial Paragraph):
    \"\"\"{initial_paragraph}\"\"\"

    Passage 2 (Story):
    \"\"\"{story_text}\"\"\"

    and give a match score on a scale from 0 to 1, where:
    - 0 means completely unrelated
    - 1 means the stories have the same theme and very similar events
    
    Output ONLY the compatibility score as a decimal number (e.g., 0.75). 
    No extra text. 
    """

    full_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    command = [
        LLAMA_CLI_PATH,
        "-m", MODEL_PATH,
        "-c", "2048",
        "-n", "50",
        "--temp", "0.7",
        "--top-p", "0.9",
        "--seed", "42",
        "--prompt", full_prompt
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout

        # Extract assistant response
        assistant_marker = "assistant\n\n"
        start = output.rfind(assistant_marker)
        if start != -1:
            response = output[start + len(assistant_marker):].strip()
        else:
            response = output.strip()

        # Clean response from tokens
        response = response.replace("<|eot_id|>", "").replace("[end of text]", "").strip()

        # Find the first decimal number in the response
        match = re.search(r"0(\.\d+)?|1(\.0+)?", response)
        if match:
            score = float(match.group())
            return score
        else:
            print(f"Warning: Could not parse a score from model output:\n{response}")
            return 0.0

    except subprocess.CalledProcessError as e:
        print("Error running llama-cli:", e)
        return 0.0

def find_compatible_story_with_threshold(initial_paragraph, story_file_path, threshold=0.6):
    stories = load_stories(story_file_path)
    if not stories:
        print("No stories loaded.")
        return None, 0.0, None

    best_score = -1.0
    best_story = None
    story_num = None

    for idx, story in enumerate(stories):
        print(f"Scoring story #{idx+1}/{len(stories)}...")
        score = ask_compatibility_score(initial_paragraph, story)
        print(f"Score: {score}")

        if score > best_score:
            best_score = score
            best_story = story
            story_num = idx + 1

        if score >= threshold:
            print(f"Score {score} exceeds threshold {threshold}. Selecting this story early.")
            return best_story, best_score, story_num

    print(f"No story exceeded the threshold {threshold}. Best score was {best_score}")
    return best_story, best_score, story_num

def extract_story(file_path, story_number):
    """
    Extracts the full story text from a file given the story number,
    assuming the file uses tags like --- Story X --- to separate stories.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    story_tag = f"--- Story {story_number} ---"
    next_story_tag = f"--- Story {story_number + 1} ---"

    story_lines = []
    inside_story = False

    for line in lines:
        if line.strip() == story_tag:
            inside_story = True
            continue
        elif line.strip() == next_story_tag and inside_story:
            break
        elif inside_story:
            story_lines.append(line)

    return ''.join(story_lines).strip()


def run_match():
    # Settings
    paragraph_outputs_file = f"{SG_PATH}/generated_story_outline.txt"
    story_file = f"{SG_PATH}/filtered_stories_150_to_400_AFP.txt"
    corpus_file = f"{SG_PATH}/paragraph_transitions_Glot150.txt"
    OUTPUT_FILE = f"{SG_PATH}/selected_story.txt"
    threshold = 0.6
    NUM_MATCHES = 1 # <-- Set how many matches to attempt

    # Load and sample paragraphs
    paragraphs = parse_paragraph_outputs(paragraph_outputs_file)
    selected_paragraphs = []
    for paragraph in paragraphs:
        if len(selected_paragraphs) >= NUM_MATCHES:
            break
        selected_paragraphs.append(paragraph)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        for i, (selected_story_num, initial_paragraph) in enumerate(selected_paragraphs, start=1):
            print(f"\n[{i}/{NUM_MATCHES}] Matching for paragraph from Story #{selected_story_num}:\n{initial_paragraph}\n")

            chosen_story, score, matched_story_num = find_compatible_story_with_threshold(
                initial_paragraph,
                story_file,
                threshold
            )

            if chosen_story:
                print(f"\n--- Selected Story ---\nCompatibility Score: {score:.2f}")
                print(chosen_story)

                full_story_text = extract_story(corpus_file, matched_story_num)

                # output_file.write(f"### Match {i}: Initial Paragraph from Story {selected_story_num}\n")
                # output_file.write(initial_paragraph + "\n\n")
                output_file.write(f"--- Matched Story {matched_story_num} (Score: {score:.2f}) ---\n")
                output_file.write(full_story_text + "\n\n")
            else:
                print("No story matched for this paragraph.")
                output_file.write(f"### Match {i}: Initial Paragraph from Story {selected_story_num}\n")
                output_file.write(initial_paragraph + "\n\n")
                output_file.write("No compatible story found.\n\n")

    print(f"\nAll {NUM_MATCHES} matches processed. Results saved to {OUTPUT_FILE}")

