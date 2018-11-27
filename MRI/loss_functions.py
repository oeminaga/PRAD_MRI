from keras import backend as K
from keras import losses
#custom loss function


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def combined_loss_function(weights_iou, weights_entropy, weight_ratio=[0.5,1]):
    weight_ratio_iou = K.variable(weight_ratio[0])
    weight_ratio_entropy = K.variable(weight_ratio[1])
    weights_iou = K.variable(weights_iou)
    weights_entropy = K.variable(weights_entropy)
    
    def loss_iou(y_true, y_pred, smooth=100):
        intersection = K.sum(K.abs(y_true * y_pred * weights_iou), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        loss = (1 - jac) * smooth
        return loss
    
    def loss_entropy(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights_entropy
        loss = -K.sum(loss, -1)
        return loss

    def loss(y_true, y_pred):
        loss = loss_iou(y_true, y_pred) * weight_ratio_iou + \
            loss_entropy(y_true, y_pred)*weight_ratio_entropy
        return loss

    return loss

def weighted_loss_IOU(weights):
    """
    Weighted loss IOU
    """
    weights = K.variable(weights)

    def loss(y_true, y_pred, smooth=100):
        
        intersection = K.sum(K.abs(y_true * y_pred * weights), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        loss = (1 - jac) * smooth
        return loss
    return loss

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)
    

def loss_IOU(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    loss = (1 - jac) * smooth
    return loss

#def binary_crossentropy(y_true, y_pred):
#return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
def weighted_binary_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred * weights), axis=-1)
    return loss

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        #loss_v = y_true * K.log(y_pred) * K.variable([0,1,1])
        #loss = (1-loss_h) * loss_v
        loss = -K.sum(loss, -1)
        return loss
    '''
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    '''
    return loss

def ignore_unknown_xentropy(ytrue, ypred):
    return (1-ytrue[:, :, :, 0])*losses.categorical_crossentropy(ytrue, ypred)
