from datasets import config
from datasets import load_dataset

config.HF_DATASETS_CACHE = "./dataset"

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("titic/MM-OPERA")

# Example of an RIA instance
ria_example = ds["ria"][0]
print(ria_example)

"""
{'foldername': 'Easter or chocolate bunny(invisible, culture, relation, USAEnglish, English)', 'image1': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=223x226 at 0x264417CE110>, 'image2': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=300x168 at 0x26441BCFF90>, 'image3': None, 'image4': None, 'relation': 'Easter or chocolate bunny', 'domain': 'culture', 'type': 'relation', 'culture': 'USA/English', 'language': 'English', 'explanation': 'The first image shows a bunny, a common symbol associated with Easter. The second image depicts chocolate, which is often given as a gift or enjoyed during Easter celebrations. Chocolate bunnies are a popular treat during this holiday, combining the themes of Easter and sweet indulgence. Therefore, the chocolate bunny serves as a link between the bunny and the chocolate.', 'hop_count': 2, 'reasoning': 'Related(Bunny, Easter)\nRelated(Chocolate, Easter)\nThus, Bunny → Easter and Chocolate → Easter', 'perception': 'Contextual Sensory Cues, Scene Contextualization', 'conception': 'Cultural Reference', 'img_id1': '1', 'filename1': 'bunny.jpg', 'description1': 'bunny', 'image_path1': 'images/RIA/Easter or chocolate bunny(invisible, culture, relation, USAEnglish, English)/bunny.jpg', 'img_id2': '2', 'filename2': 'chocolate.jpg', 'description2': 'chocolate', 'image_path2': 'images/RIA/Easter or chocolate bunny(invisible, culture, relation, USAEnglish, English)/chocolate.jpg', 'img_id3': None, 'filename3': None, 'description3': None, 'image_path3': None, 'img_id4': None, 'filename4': None, 'description4': None, 'image_path4': None}
"""

# Example of an ICA instance
ica_example = ds["ica"][0]
print(ica_example)

"""
{'foldername': 'রাষ্ট্র চিহ্নের প্রাণী(animal, sports, relation, USAEnglish, Bengali)', 'image1': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=309x163 at 0x264415D1990>, 'image2': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=275x183 at 0x264415D1CD0>, 'image3': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=275x183 at 0x264415D2050>, 'image4': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=300x168 at 0x264415D23D0>, 'relation': 'রাষ্ট্র চিহ্নের প্রাণী', 'domain': 'animal, sports', 'type': 'relation', 'culture': 'USA/English', 'language': 'Bengali', 'explanation': None, 'hop_count': 2, 'reasoning': '[{"pair_id": "1", "explanation": "The bald eagle is the symbol on the national emblem of the United States, where the basketball originated.", "path": "NationalEmblem(BaldEagle, US) and Origin(US, Basketball)\\nThus, BaldEagle \\u2192 US \\u2192 Basketball", "perception": ["Scene Contextualization", "Semantic Object", "Relational Perception"], "conception": ["Analogical Reasoning", "Cultural Reference"]}, {"pair_id": "2", "explanation": "The lion is the symbol on the national emblem of England, where the football originated.", "path": "NationalEmblem(Lion, England) and Origin(England, Football)\\nThus, Lion \\u2192 England \\u2192 Football", "perception": ["Relational Perception", "Scene Contextualization", "Semantic Object"], "conception": ["Cultural Reference", "Analogical Reasoning"]}]', 'perception': None, 'conception': None, 'img_id1': '1', 'filename1': 'batle.jpg', 'description1': 'A bald eagle', 'image_path1': 'images/ICA/রাষ্ট্র চিহ্নের প্রাণী(animal, sports, relation, USAEnglish, Bengali)/batle.jpg', 'img_id2': '2', 'filename2': 'baske.jpg', 'description2': 'Basketball game', 'image_path2': 'images/ICA/রাষ্ট্র চিহ্নের প্রাণী(animal, sports, relation, USAEnglish, Bengali)/baske.jpg', 'img_id3': '3', 'filename3': 'lions.jpg', 'description3': ' A lion', 'image_path3': 'images/ICA/রাষ্ট্র চিহ্নের প্রাণী(animal, sports, relation, USAEnglish, Bengali)/lions.jpg', 'img_id4': '4', 'filename4': 'soccer.jpg', 'description4': 'Football game', 'image_path4': 'images/ICA/রাষ্ট্র চিহ্নের প্রাণী(animal, sports, relation, USAEnglish, Bengali)/soccer.jpg'}
"""
