from tensorflow.keras import Model
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU,concatenate
#from tensorflow.math import square


class DnCNN(Model):
    def __init__(self, depth=17):
        super(DnCNN, self).__init__()

        # Initial conv + relu
        self.conv1 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer=he_uniform())

        # Depth - 2 cnv+bn+relu layers
        self.conv_bn_relu = [ConvBNReLU() for i in range(depth - 2)]

        # final conv
        self.conv_final = Conv2D(1, 3, padding='same', kernel_initializer=he_uniform())

    def call(self, x):
        out = self.conv1(x)
        for cbr in self.conv_bn_relu:
            out = cbr(out)
        out1=self.conv1(out)
        out2=self.conv1(out)
        out3=concatenate(out1,out2)
        
        return x - self.conv_final(out1),x - self.conv_final(out2),x - self.conv_final(out3)


class ConvBNReLU(Model):
    def __init__(self):
        super(ConvBNReLU, self).__init__()
        self.conv = Conv2D(64, 3, padding='same', kernel_initializer=he_uniform())
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


