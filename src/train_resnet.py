import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import torchvision
from tqdm import tqdm
from typing import Type, Optional
from dataclasses import dataclass
from utils import get_resnet_for_fine_tuning
import wandb
import os
from sklearn.metrics import precision_recall_fscore_support, auc, roc_curve, precision_recall_curve

device = 'cuda' if t.cuda.is_available() else 'cpu'

def compute_auroc(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

def compute_auprc(y_true, y_score) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

@dataclass
class TrainingArgs:
    # img_path: str = "../music/images"
    img_train_path: str = "../dataset/images/train"
    img_test_path: str = "../dataset/images/test"
    batch_size: int = 32
    num_classes: int = 2
    epochs: int = 20
    lr: float = 1e-3
    model_name: str = 'TFBS_resnet50.pth'
    outdir: str = '../models'
    wandb_project: Optional[str] = 'TFBS-resnet'
    wandb_name: Optional[str] = None    

class ResNetTrainer:

    def __init__(self, args: TrainingArgs):
        self.args = args
        self.model = get_resnet_for_fine_tuning(self.args.num_classes).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr = self.args.lr)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.486, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # self.images = torchvision.datasets.ImageFolder(self.args.img_path, transform=self.transform)
        # self.train_data, self.test_data = data.random_split(self.images, [0.9, 0.1])
        self.train_data = torchvision.datasets.ImageFolder(self.args.img_train_path, transform=self.transform)
        self.test_data = torchvision.datasets.ImageFolder(self.args.img_test_path, transform=self.transform)
        self.step = 0
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
        wandb.watch(self.model.fc, log='all', log_freq=20)

    def train_dataloader(self):
        return data.DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return data.DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=True)

    def _shared_train_val_step(self, imgs, labels):
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = self.model(imgs)
        return logits, labels
    
    def training_step(self, imgs, labels):
        logits, labels = self._shared_train_val_step(imgs, labels)
        loss = self.loss_fn(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        return loss
    
    @t.inference_mode
    def val_step(self, imgs, labels):
        logits, labels = self._shared_train_val_step(imgs, labels) 
        probabilities = F.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1) 
        # correct = t.sum(predictions == labels)
        return predictions.cpu(), probabilities.cpu(), labels.cpu()

    
    def training_loop(self):
        
        self.model.train()
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        progress_bar = tqdm(total = self.args.epochs * len(self.train_data) // self.args.batch_size)
        accuracy = t.nan
        best_accuracy = 0

        for epoch in range(self.args.epochs):
            for imgs, labels in self.train_dataloader():
                loss = self.training_step(imgs, labels)
                wandb.log({'loss': loss.item()}, step=self.step)
                progress_bar.set_description(f"Epoch: {epoch}, Loss: {loss}")
                progress_bar.update()

            # Validation step
            all_predictions, all_probabilities, all_labels = [], [], [] 
            for imgs, labels in self.val_dataloader():
                preds, probs, lbls = self.val_step(imgs, labels)
                all_predictions.append(preds)
                all_probabilities.append(probs)
                all_labels.append(lbls)

            all_predictions = t.cat(all_predictions).numpy()
            all_probabilities = t.cat(all_probabilities).numpy()
            all_labels = t.cat(all_labels).numpy()

            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary', zero_division=0)
            auroc = compute_auroc(all_labels, all_probabilities)
            auprc = compute_auprc(all_labels, all_probabilities)
            accuracy = (all_labels == all_predictions).mean()

            # accuracy = sum(self.val_step(imgs, labels) for imgs, labels in self.val_dataloader()) / len(self.test_data)
            # wandb.log({'accuracy': accuracy}, step=self.step)
            wandb.log({"accuracy": accuracy,
                       "precision": precision,
                       "recall": recall,
                       "f1_score": f1,
                       "auroc": auroc,
                       "auprc": auprc}, step = self.step)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                t.save(self.model.state_dict(), os.path.join(self.args.outdir, self.args.model_name))
                print(f"Model saved in {self.args.outdir} with accuracy {accuracy}")

        wandb.finish()

def main():
    args = TrainingArgs()
    trainer = ResNetTrainer(args)
    trainer.training_loop()

if __name__ == '__main__':
    main()
