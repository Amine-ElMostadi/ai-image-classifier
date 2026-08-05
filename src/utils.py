def accuracy(predictions, labels):
    """Calculate accuracy"""
    correct = (predictions.argmax(dim=1) == labels).sum().item()
    return correct / len(labels)
