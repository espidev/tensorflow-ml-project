import inputs
import model_tools as mt
import grayscale_model as gm
import colour_model as cm

# Input from file & Data Augmentation

# inputs.serialize()
# inputs.augment()
# inputs.grayscale()

# Neural Network for Colour Images 150x150

# grayscale_model = gm.get_model()
# grayscale_model.summary()
# gm.run(grayscale_model, plot=True, test=True, save=True)

# Neural Network for Colour Images 150x150

colour_model = cm.get_model()
colour_model.summary()
cm.run(colour_model, mix=True, plot=False, test=True, save=True)

aug_model = cm.get_model()
aug_model.summary()
cm.run(aug_model, mix=False, plot=False, test=True, save=True)
# # use mix = True to train on both augmented and raw data

# colour_model = modelIO.load_model("colour_model")
