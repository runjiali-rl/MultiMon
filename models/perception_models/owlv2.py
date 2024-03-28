import requests
from PIL import Image
import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection


class Owlv2:
    def __init__(self, cache_dir=None, device='cuda'):
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16",
                                                        cache_dir=cache_dir,
                                                        device=device)
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16",
                                                             cache_dir=cache_dir)
        self.model.to(device)
        self.model = self.model.half()

    def detect(self, texts, image, threshold=0.5):
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        # convert inputs to device
        inputs = {name: tensor.to(self.model.device) for name, tensor in inputs.items()}
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs,
                                                               target_sizes=target_sizes,
                                                               threshold=threshold)[0]
        # del inputs
        # del outputs
        # del target_sizes
        return results

# processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
# model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# texts = [["a photo of a cat", "a photo of a dog"]]
# inputs = processor(text=texts, images=image, return_tensors="pt")
# outputs = model(**inputs)

# # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
# target_sizes = torch.Tensor([image.size[::-1]])
# # Convert outputs (bounding boxes and class logits) to COCO API
# results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

# i = 0  # Retrieve predictions for the first image for the corresponding text queries
# text = texts[i]
# boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# # Print detected objects and rescaled box coordinates
# for box, score, label in zip(boxes, scores, labels):
#     box = [round(i, 2) for i in box.tolist()]
#     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")


if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    texts = [["a photo of a cat", "a photo of a dog"]]

    owlv2 = Owlv2(cache_dir="/homes/55/runjia/scratch/diffusion_weights")
    results = owlv2.detect(texts, image)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {owlv2.model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

    stop = 1