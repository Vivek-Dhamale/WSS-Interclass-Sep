import torch
import torch.optim as optim
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix

from models import *
from loss import *
from utils import *

class CNet(object):
    def __init__(self, args, train_loader, val_loader):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.project_path = args.project_path
        self.record_path = args.record_path
        self.model_name = args.model_name
        self.task = args.task  # 'binary' or 'multiclass'
        self.lr = args.learning_rate

        model = Res18_Classifier(num_classes=self.num_classes)

        if args.encoder_pretrained_path is not None:
            model.load_encoder_pretrain_weight(args.encoder_pretrained_path)

        if args.pretrained_path is not None:
            model.load_pretrain_weight(args.pretrained_path)

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=5e-6)

        if len(args.gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=args.gpuid)

        self.model = model.to('cuda')
        self.loss = torch.nn.BCEWithLogitsLoss() if self.task == 'binary' else MultiLabelFocalLoss()

    def run(self):

        # Define save directory
        save_dir = os.path.join(self.project_path, self.record_path, f"{self.task}_clstrain")
        os.makedirs(save_dir, exist_ok=True)

        log_path = os.path.join(save_dir, "log.log")
        model_name = "binary_classifier.pth" if self.task == "binary" else "multi_classifier.pth"
        model_path = os.path.join(save_dir, model_name)


        with open(log_path, "w+") as log_file:
            log_file.writelines(str(datetime.now()) + "\n")

        train_record = {'auc': [], 'loss': []}
        val_record = {'auc': [], 'loss': []}
        best_score = 0.0

        val_interval = 2
        best_epoch = -1

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc, train_hard_acc, sensitivity, specificity, train_auc = self.train(epoch)
            train_record['loss'].append(train_loss)
            train_record['auc'].append(train_auc)
            
            self.scheduler.step()

            with open(log_path, "a") as log_file:
                log_file.writelines(
                    f'Epoch {epoch:4d}/{self.epochs:4d} | Cur lr: {self.scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, AUC: {train_auc:.4f}, Hard Acc: {train_hard_acc:.4f}\n'
                )
            
            if epoch % val_interval == 0:

                val_loss, val_acc, val_hard_acc, sensitivity, specificity, val_auc = self.val(epoch)
                val_record['loss'].append(val_loss)
                val_record['auc'].append(val_auc)

                with open(log_path, "a") as log_file:
                    log_file.writelines(
                        f"Epoch {epoch:4d}/{self.epochs:4d} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, Hard Acc: {val_hard_acc:.4f}, Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}\n"
                    )

                if val_auc > best_score:
                    best_score = val_auc
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), model_path)
                    with open(log_path, "a") as log_file:
                        log_file.write(f">>> Saved best model at epoch {epoch} with val AUC {val_auc:.4f}\n")
    
            with open(log_path, "a") as log_file:   
                log_file.writelines(str(datetime.now()) + "\n")

            plot_path = os.path.join(save_dir, "loss_curve.png")
            plot_loss(train_record['loss'], val_record['loss'], val_interval=val_interval, save_path=plot_path)

            plot_path = os.path.join(save_dir, "auc_curve.png")
            plot_loss(train_record['auc'], val_record['auc'], val_interval=val_interval, save_path=plot_path)

        # Final log message
        with open(log_path, "a") as log_file:
            log_file.write(f"Best model saved at epoch {best_epoch} with val AUC {best_score:.4f}\n")
        

    def train(self, epoch):
        self.model.train()
        train_bar = tqdm(self.train_loader)
        total_loss, total_num = 0.0, 0
        train_labels, pred_results = [], []
        log_path = os.path.join(self.project_path, self.record_path, f"{self.task}_clstrain", "log.log")
        with open(log_path, "a") as log_file:
            for case_batch, label_batch in train_bar:
                self.optimizer.zero_grad()
                loss, pred_batch, loss_collect = self.step(case_batch, label_batch)
                loss.backward()
                self.optimizer.step()

                total_num += case_batch.size(0)
                total_loss += loss.item() * case_batch.size(0)
                train_bar.set_description(f'Train Epoch: [{epoch}/{self.epochs}] Loss: {total_loss / total_num:.4f}')

                pred_results.append(pred_batch)
                train_labels.append(label_batch.detach().cpu())
                log_file.writelines(f'Train Loss (per-head): {[round(l.item(), 4) for l in loss_collect]}\n')

        pred_results = torch.cat(pred_results, dim=0).numpy()
        train_labels = torch.cat(train_labels, dim=0).numpy()
        acc, hard_accuracy, sensitivity, specificity, auc_score = self.evaluate(train_labels, pred_results)
        return total_loss / total_num, acc, hard_accuracy, sensitivity, specificity, auc_score

    def val(self, epoch):
        self.model.eval()
        val_bar = tqdm(self.val_loader)
        total_loss, total_num = 0.0, 0
        val_labels, pred_results = [], []

        with torch.no_grad():
            for case_batch, label_batch in val_bar:
                loss, pred_batch, _ = self.step(case_batch, label_batch)
                total_num += case_batch.size(0)
                total_loss += loss.item() * case_batch.size(0)
                val_bar.set_description(f'Val Epoch: [{epoch}/{self.epochs}] Loss: {total_loss / total_num:.4f}')
                pred_results.append(pred_batch)
                val_labels.append(label_batch.detach().cpu())

        pred_results = torch.cat(pred_results, dim=0).numpy()
        val_labels = torch.cat(val_labels, dim=0).numpy()
        acc, hard_accuracy, sensitivity, specificity, auc_score = self.evaluate(val_labels, pred_results)
        return total_loss / total_num, acc, hard_accuracy, sensitivity, specificity, auc_score

    def step(self, data_batch, label_batch, no_cl=False):
        logits_collect, _,= self.model(data_batch.cuda())
        loss = 0
        ic_weight = [0.25, 0.5, 0.75, 1.0]
        loss_collect = []

        for idx, logits in enumerate(logits_collect):
            loss_val = self.loss(logits, label_batch.cuda().float())
            loss_collect.append(loss_val.detach().cpu())
            loss += ic_weight[idx] * loss_val

        pred = torch.sigmoid(logits_collect[-1])
        return loss, pred.detach().cpu(), loss_collect

    def evaluate(self, labels, pred):
        out_results = [p > 0.5 for p in pred]
        
        if labels.shape[1] > 1:
        # Multi-class classification case
            auc_score = roc_auc_score(labels, pred, average='macro', multi_class='ovr')
        else:
        # Binary classification case
            auc_score = roc_auc_score(labels, pred, average='macro')

        cm = multilabel_confusion_matrix(labels, out_results)
        tn, fp, fn, tp = cm['tn'], cm['fp'], cm['fn'], cm['tp']
        acc = (tp + tn) / (tn + fp + fn + tp + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        sensitivity = tp / (tp + fn + 1e-7)
        hard_accuracy = multilabel_hard_accuracy(labels, out_results)
        return acc, hard_accuracy, sensitivity, specificity, auc_score


def multilabel_confusion_matrix(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    n_classes = y_true.shape[1]
    results = {'tn': [], 'fp': [], 'fn': [], 'tp': []}

    for i in range(n_classes):
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()
        results['tn'].append(tn)
        results['fp'].append(fp)
        results['fn'].append(fn)
        results['tp'].append(tp)

    results['tn'] = np.sum(results['tn'])
    results['fp'] = np.sum(results['fp'])
    results['fn'] = np.sum(results['fn'])
    results['tp'] = np.sum(results['tp'])

    return results


def multilabel_hard_accuracy(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    correct = sum(np.array_equal(y_true[i], y_pred[i]) for i in range(y_true.shape[0]))
    return correct / y_true.shape[0]
