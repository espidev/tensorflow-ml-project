import inputs
import model_tools as mt
import grayscale_model as gm
import colour_model as cm
import vgg19_model as vgg

# Input from file & Data Augmentation

# inputs.serialize()
# inputs.augment()
# inputs.grayscale()
# inputs.format_vgg()
# Neural Network for Colour Images 150x150

# grayscale_model = gm.get_model()
# grayscale_model.summary()
# gm.run(grayscale_model, plot=True, test=True, save=False)

# Neural Network for Colour Images 150x150

# colour_model = cm.get_model()
# colour_model.summary()
# cm.run(colour_model, mix=True, plot=False, test=True, save=True)

# aug_model = cm.get_model()
# aug_model.summary()
# cm.run(aug_model, mix=False, plot=False, test=True, save=True)
# # use mix = True to train on both augmented and raw data

# colour_model = mt.load_model("colour_model")

# VGG19 Transfer Network for Colour Images 224x224
# vgg.imagenet()
# vgg.vgg_conv()
# vgg_top = vgg.get_model()
# vgg.run(vgg_top, plot=True, test=True, save=True)

vgg_model = mt.load_model("topVGG19model")
vgg.conf_matrix(vgg_model)

# Large-Scale Predicting

# vgg.grid("sample01.tif")
# vgg.grid("sample02.tif")
# vgg.grid("sample03.tif")
