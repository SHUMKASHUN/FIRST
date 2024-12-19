import jsonlines
import random
from torch.utils.data import Dataset

class TeacherDataset(Dataset):
    def __init__(self, data_path: str, percentage: float = 1):
        self.data = []
        with open(data_path, "r") as f:
            for item in jsonlines.Reader(f): 
                self.data.append(item)
        random.shuffle(self.data)  # shuffle the data after loading
        bound = round(len(self.data) * percentage)
        self.data = self.data[:bound]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return{
            'text': self.data[index]['text'],
            'top_token_prob': self.data[index]['top_token_prob'],
            'loss_mask': self.data[index]['loss_mask'],
            'attention_mask': self.data[index]['attention_mask']
        }
    
    def collate_fn(self, batch):
        return{
            'input_token': [x['text'] for x in batch],
            'output_token': [[[int(z) for z in list(y.keys())] for y in x['top_token_prob']] for x in batch],
            'top_prob': [[list(y.values()) for y in x['top_token_prob']] for x in batch],
            "loss_mask": [x['loss_mask'] for x in batch],
            "attention_mask": [x["attention_mask"] for x in batch]
        }
