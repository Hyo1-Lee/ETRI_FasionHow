from edafa import ClassPredictor

class myPredictor(ClassPredictor):
    def __init__(self,model,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = model

    def predict_patches(self,patches):
        return self.model.predict(patches)
    

# conf = '{"augs":["NO",\
#                 "FLIP_LR"],\
#         "mean":"ARITH"}'
        
# p = myPredictor(model,conf)

# y_pred_aug = p.predict_images(X_val)

# y_pred_aug = [(y[0]>=0.5).astype(np.uint8) for y in y_pred_aug ]

# print('Accuracy with TTA:',np.mean((y_val==y_pred_aug)))
