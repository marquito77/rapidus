from configparser import ConfigParser
from collections import OrderedDict
import logging
import os
import numpy as np

os.environ['GLOG_minloglevel'] = '2' 
import caffe

class CaffeLayerGenerator(object):
    def __init__(self, name, ltype):
        self.name = name
        self.bottom = []
        self.top = []
        self.type = ltype
    def get_template(self):
        return """
layer {{{{
  name: "{}"
  type: "{}"
  bottom: "{}"
  top: "{}"{{}}
}}}}""".format(self.name, self.type, self.bottom[0], self.top[0])

class CaffeInputLayer(CaffeLayerGenerator):
    def __init__(self, name, channels, width, height):
        super(CaffeInputLayer, self).__init__(name, 'Input')
        self.channels = channels
        self.width = width
        self.height = height
    def write(self, f):
        f.write("""
input: "{}"
input_shape {{
  dim: 1
  dim: {}
  dim: {}
  dim: {}
}}""".format(self.name, self.channels, self.width, self.height))

class CaffeConvolutionLayer(CaffeLayerGenerator):
    def __init__(self, name, filters, ksize=None, stride=None, pad=None, bias=True):
        super(CaffeConvolutionLayer, self).__init__(name, 'Convolution')
        self.filters = filters
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.bias = bias
    def write(self, f):
        opts = ['']
        if self.ksize is not None: opts.append('kernel_size: {}'.format(self.ksize))
        if self.stride is not None: opts.append('stride: {}'.format(self.stride))
        if self.pad is not None: opts.append('pad: {}'.format(self.pad))
        if not self.bias: opts.append('bias_term: false')
        param_str = """
  convolution_param {{
    num_output: {}{}
  }}""".format(self.filters, '\n    '.join(opts))
        f.write(self.get_template().format(param_str))

class CaffePoolingLayer(CaffeLayerGenerator):
    def __init__(self, name, pooltype, ksize=None, stride=None, pad=None, global_pooling=None):
        super(CaffePoolingLayer, self).__init__(name, 'Pooling')
        self.pooltype = pooltype
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.global_pooling = global_pooling
    def write(self, f):
        opts = ['']
        if self.ksize is not None: opts.append('kernel_size: {}'.format(self.ksize))
        if self.stride is not None: opts.append('stride: {}'.format(self.stride))
        if self.pad is not None: opts.append('pad: {}'.format(self.pad))
        if self.global_pooling is not None: opts.append('global_pooling: {}'.format('True' if self.global_pooling else 'False'))
        param_str = """
  pooling_param {{
    pool: {}{}
  }}""".format(self.pooltype, '\n    '.join(opts))
        f.write(self.get_template().format(param_str))

class CaffeInnerProductLayer(CaffeLayerGenerator):
    def __init__(self, name, num_output):
        super(CaffeInnerProductLayer, self).__init__(name, 'InnerProduct')
        self.num_output = num_output
    def write(self, f):
        param_str = """
  inner_product_param {{
    num_output: {}
  }}""".format(self.num_output)
        f.write(self.get_template().format(param_str))

class CaffeBatchNormLayer(CaffeLayerGenerator):
    def __init__(self, name):
        super(CaffeBatchNormLayer, self).__init__(name, 'BatchNorm')
    def write(self, f):
        param_str = """
  batch_norm_param {
    use_global_stats: true
  }"""
        f.write(self.get_template().format(param_str))

class CaffeScaleLayer(CaffeLayerGenerator):
    def __init__(self, name):
        super(CaffeScaleLayer, self).__init__(name, 'Scale')
    def write(self, f):
        param_str = """
  scale_param {
    bias_term: true
  }"""
        f.write(self.get_template().format(param_str))

class CaffeReluLayer(CaffeLayerGenerator):
    def __init__(self, name, negslope=None):
        super(CaffeReluLayer, self).__init__(name, 'ReLU')
        self.negslope = negslope
    def write(self, f):
        param_str = ""
        if self.negslope is not None:
            param_str = """
  relu_param {{
    negative_slope: {}
  }}""".format(self.negslope)
        f.write(self.get_template().format(param_str))

class CaffeDropoutLayer(CaffeLayerGenerator):
    def __init__(self, name, prob):
        super(CaffeDropoutLayer, self).__init__(name, 'Dropout')
        self.prob = prob
    def write(self, f):
        param_str = """
  dropout_param {{
    dropout_ratio: {}
  }}""".format(self.prob)
        f.write(self.get_template().format(param_str))

class CaffeSoftmaxLayer(CaffeLayerGenerator):
    def __init__(self, name):
        super(CaffeSoftmaxLayer, self).__init__(name, 'Softmax')
    def write(self, f):
        f.write(self.get_template().format(""))

class CaffeProtoGenerator:
    def __init__(self, name):
        self.name = name
        self.sections = []
        self.lnum = 0
        self.layer = None
    def add_layer(self, l):
        self.sections.append( l )
    def add_input_layer(self, items):
        self.lnum = 0
        lname = "data"
        self.layer = CaffeInputLayer(lname, items['channels'], items['width'], items['height'])
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def update_last_convolution_layer(self):
        self.sections[len(self.sections)-1].pad = 0
    def add_convolution_layer(self, items):
        self.lnum += 1
        prev_blob = self.layer.top[0]
        lname = "conv"+str(self.lnum)
        filters = items['filters']
        ksize = items['size'] if 'size' in items else None
        stride = items['stride'] if 'stride' in items else None
        pad = items['pad'] if 'pad' in items else None
        bias = not bool(items['batch_normalize']) if 'batch_normalize' in items else True
        self.layer = CaffeConvolutionLayer( lname, filters, ksize=ksize, stride=stride, pad=pad, bias=bias )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def add_innerproduct_layer(self, items):
        self.lnum += 1
        prev_blob = self.layer.top[0]
        lname = "fc"+str(self.lnum)
        num_output = items['output']
        self.layer = CaffeInnerProductLayer( lname, num_output )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def add_pooling_layer(self, ltype, items, global_pooling=None):
        prev_blob = self.layer.top[0]
        lname = "pool"+str(self.lnum)
        ksize = items['size'] if 'size' in items else None
        stride = items['stride'] if 'stride' in items else None
        pad = items['pad'] if 'pad' in items else None
        self.layer = CaffePoolingLayer( lname, ltype, ksize=ksize, stride=stride, pad=pad, global_pooling=global_pooling )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def add_batchnorm_layer(self, items):
        prev_blob = self.layer.top[0]
        lname = "bn"+str(self.lnum)
        self.layer = CaffeBatchNormLayer( lname )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def add_scale_layer(self, items):
        prev_blob = self.layer.top[0]
        lname = "scale"+str(self.lnum)
        self.layer = CaffeScaleLayer( lname )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def add_relu_layer(self, items):
        prev_blob = self.layer.top[0]
        lname = "relu"+str(self.lnum)
        if items['activation'] == "relu":
            self.layer = CaffeReluLayer( lname )
        elif items['activation'] == "leaky":
            self.layer = CaffeReluLayer( lname, 0.1 )
        else:
            print("Unknown activation: {}".format(items['activation']))
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( prev_blob )     # loopback
        self.add_layer( self.layer )
    def add_dropout_layer(self, items):
        prev_blob = self.layer.top[0]
        lname = "drop"+str(self.lnum)
        self.layer = CaffeDropoutLayer( lname, items['probability'] )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( prev_blob )     # loopback
        self.add_layer( self.layer )
    def add_softmax_layer(self, items):
        prev_blob = self.layer.top[0]
        lname = "prob"
        self.layer = CaffeSoftmaxLayer( lname )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def finalize(self, name):
        self.layer.top[0] = name    # replace
    def write(self, fname):
        with open(fname, 'w') as f:
            f.write('name: "{}"'.format(self.name))
            for sec in self.sections:
                sec.write(f)
        logging.info('{} is generated'.format(fname))

###################################################################33
class uniqdict(OrderedDict):
    _unique = 0
    def __setitem__(self, key, val):
        if isinstance(val, OrderedDict):
            self._unique += 1
            key += "_"+str(self._unique)
        OrderedDict.__setitem__(self, key, val)

def convertCfgToPrototxt(cfgfile, targetDir):
    
    if targetDir == None:
        targetDir = os.path.dirname(cfgfile)
    targetDir = os.path.abspath(targetDir)
    
    cfgfile = os.path.abspath(cfgfile)
    
    ptxtfile = os.path.basename(cfgfile)
    ptxtfile = os.path.splitext(ptxtfile)[0]
    ptxtfile += ".prototxt"
    ptxtfile = os.path.join(targetDir, ptxtfile)
        
    parser = ConfigParser(dict_type=uniqdict, strict=False)
    parser.read(cfgfile)
    netname = os.path.basename(cfgfile).split('.')[0]
    #print netname
    gen = CaffeProtoGenerator(netname)
    for section in parser.sections():
        _section = section.split('_')[0]
        if _section in ["crop", "cost"]:
            continue
        #
        batchnorm_followed = False
        relu_followed = False
        items = dict(parser.items(section))
        if 'batch_normalize' in items and items['batch_normalize']:
            batchnorm_followed = True
        if 'activation' in items and items['activation'] != 'linear':
            relu_followed = True
        #
        if _section == 'net':
            gen.add_input_layer(items)
        elif _section == 'convolutional':
            gen.add_convolution_layer(items)
            if batchnorm_followed:
                gen.add_batchnorm_layer(items)
                gen.add_scale_layer(items)
            if relu_followed:
                gen.add_relu_layer(items)
        elif _section == 'connected':
            gen.add_innerproduct_layer(items)
            if relu_followed:
                gen.add_relu_layer(items)
        elif _section == 'maxpool':
            gen.add_pooling_layer('MAX', items)
        elif _section == 'avgpool':
            gen.add_pooling_layer('AVE', items, global_pooling=True)
        elif _section == 'dropout':
            gen.add_dropout_layer(items)
        elif _section == 'softmax':
            gen.add_softmax_layer(items)
        else:
            logging.warning("{} layer is not supported".format(_section))
    gen.update_last_convolution_layer()
    #gen.finalize('result')
    gen.write(ptxtfile)
    return ptxtfile

def convertWeightsToCaffemodel(weightsFile, targetDir, prototxtFile = None):
    if targetDir == None:
        targetDir = os.path.dirname(weightsFile)
    targetDir = os.path.abspath(targetDir)
    
    if prototxtFile == None:
        fn = os.path.basename(weightsFile)
        fn = os.path.splitext(fn)[0]
        fn += ".prototxt"
        prototxtFile = os.path.join(targetDir, fn)
        
    if not os.path.isfile(prototxtFile):
        print("Could not find file {}".format(prototxtFile))
        return None
    
    caffeFile = os.path.basename(weightsFile)
    caffeFile = os.path.splitext(caffeFile)[0]
    caffeFile += ".caffemodel"
    caffeFile = os.path.join(targetDir, caffeFile)
    
    net = caffe.Net(prototxtFile, caffe.TEST)
    params = net.params.keys()
        # read weights from file and assign to the network
    netWeightsInt = np.fromfile(weightsFile, dtype=np.int32)
    transFlag = (netWeightsInt[0]>1000 or netWeightsInt[1]>1000) 
    # transpose flag, the first 4 entries are major, minor, revision and net.seen
    start = 4   
    if (netWeightsInt[0]*10 + netWeightsInt[1]) >= 2:
        start = 5


    netWeightsFloat = np.fromfile(weightsFile, dtype=np.float32)
    netWeights = netWeightsFloat[start:] # start from the 5th entry, the first 4 entries are major, minor, revision and net.seen
    count = 0

    #print("#Total Net Layer", len(net.layers))

    layercnt = 0
    for pr in params:
        layercnt = layercnt + 1
        lidx = list(net._layer_names).index(pr)
        layer = net.layers[lidx]
        if count == netWeights.shape[0] and (layer.type != 'BatchNorm' and layer.type != 'Scale'):
            print("WARNING: no weights left for %s" % pr)
            break
        if layer.type == 'Convolution':
            #print(pr+"(conv)" + "-"+str(layercnt)+"-"+str(len(net.params[pr]) > 1))
            # bias
            if len(net.params[pr]) > 1:
                bias_dim = net.params[pr][1].data.shape
            else:
                bias_dim = (net.params[pr][0].data.shape[0], )
            biasSize = np.prod(bias_dim)
            conv_bias = np.reshape(netWeights[count:count+biasSize], bias_dim)
            if len(net.params[pr]) > 1:
                assert(bias_dim == net.params[pr][1].data.shape)
                net.params[pr][1].data[...] = conv_bias
                conv_bias = None
            count = count + biasSize
            # batch_norm
            if lidx+1 < len(net.layers) and net.layers[lidx+1].type == 'BatchNorm':
                bn_dims = (3, net.params[pr][0].data.shape[0])
                bnSize = np.prod(bn_dims)
                batch_norm = np.reshape(netWeights[count:count+bnSize], bn_dims)
                count = count + bnSize
            # weights
            dims = net.params[pr][0].data.shape
            weightSize = np.prod(dims)
            net.params[pr][0].data[...] = np.reshape(netWeights[count:count+weightSize], dims)
            count = count + weightSize
            
        elif layer.type == 'InnerProduct':
            #print(pr+"(fc)")
            # bias
            biasSize = np.prod(net.params[pr][1].data.shape)
            net.params[pr][1].data[...] = np.reshape(netWeights[count:count+biasSize], net.params[pr][1].data.shape)
            count = count + biasSize
            # weights
            dims = net.params[pr][0].data.shape
            weightSize = np.prod(dims)
            if transFlag:
                net.params[pr][0].data[...] = np.reshape(netWeights[count:count+weightSize], (dims[1], dims[0])).transpose()
            else:
                net.params[pr][0].data[...] = np.reshape(netWeights[count:count+weightSize], dims)
            count = count + weightSize
        elif layer.type == 'BatchNorm':
            #print(pr+"(batchnorm)")
            net.params[pr][0].data[...] = batch_norm[1] # mean
            net.params[pr][1].data[...] = batch_norm[2] # variance
            net.params[pr][2].data[...] = 1.0   # scale factor
        elif layer.type == 'Scale':
            #print(pr+"(scale)")
            if batch_norm is not None:
                net.params[pr][0].data[...] = batch_norm[0] # scale
            batch_norm = None
            if len(net.params[pr]) > 1:
                net.params[pr][1].data[...] = conv_bias # bias
                conv_bias = None
        else:
            print("WARNING: unsupported layer, "+pr)
    if np.prod(netWeights.shape) != count:
        print("ERROR: size mismatch: %d" % count)
    net.save(caffeFile)
    return caffeFile    

    

def convertYoloToCaffe(cfgFile=None, weightsFile = None, targetDir=None):
    if (cfgFile==None) and (weightsFile==None):
        print("No yolo files specified. Nothing to do.")
        return

    if targetDir != None:
        if not os.path.isdir(targetDir):
            print("Could not find target dir {}".format(targetDir))
            return            

    prototxtFile = None
    if cfgFile != None:
        if os.path.isfile(cfgFile):
            prototxtFile = convertCfgToPrototxt(cfgFile, targetDir)
            print("Successfully created {}".format(prototxtFile))
        else:
            print("Could not find file {}".format(cfgFile))
            return
    
    if weightsFile != None:
        if os.path.isfile(weightsFile):
            fn = convertWeightsToCaffemodel(weightsFile, targetDir, prototxtFile)
            if fn != None:
                print("Successfully created {}".format(fn))
        else:
            print("Could not find file {}".format(weightsFile))
            return
            
