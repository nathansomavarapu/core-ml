from modules.BaseMLModule import BaseMLModule

from utils.metrics import correct

class VisualClassificationModule(BaseMLModule):
    
    def train(self) -> None:
        """Runs one training epoch, saving metrics to the training log.
        """
        train_loss = 0.0
        train_batches = len(self.trainloader)
        train_correct = 0.0
        train_total = 0.0

        for i,data in enumerate(self.trainloader):
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)

            pred = self.model(images)
            loss = self.loss_fn(pred, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_correct += correct(pred, labels)
            train_total += images.size(0)
        
        self.train_log = {
            'train_acc': train_correct / train_total,
            'train_loss': train_loss / train_batches
        }

    def val(self) -> None:
        """Runs one validation epoch, saving metrics to the training log.
        """
        val_loss = 0.0
        val_batches = len(self.valloader)
        val_correct = 0.0
        val_total = 0.0

        for i,data in enumerate(self.valloader):
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)

            pred = self.model(images)
            loss = self.loss_fn(pred, labels)

            val_loss += loss.item()
            val_correct += correct(pred, labels)
            val_total += images.size(0)
        
        self.val_log = {
            'val_acc': val_correct / val_total,
            'val_loss': val_loss / val_batches
        }
    
    def test(self) -> None:
        """Runs one test epoch, saving the metrics to the training log.
        """
        super(VisualClassificationModule, self).test()

        test_loss = 0.0
        test_batches = len(self.testloader)
        test_correct = 0.0
        test_total = 0.0

        for i,data in enumerate(self.testloader):
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)

            pred = self.model(images)
            loss = self.loss_fn(pred, labels)

            test_loss += loss.item()
            test_correct += correct(pred, labels)
            test_total += images.size(0)
        
        self.test_log = {
            'test_acc': test_correct / test_total,
            'test_loss': test_loss / test_batches
        }