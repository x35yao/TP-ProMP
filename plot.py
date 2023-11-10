def remove_repetitive_labels(handles, labels):
    '''
    Remove repetitive labels for Matplotlib
    '''
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    return newHandles, newLabels
