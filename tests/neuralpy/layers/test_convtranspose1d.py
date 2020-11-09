import pytest
from torch.nn import ConvTranspose1d as _ConvTranspose1d
from neuralpy.layers import ConvTranspose1d


def test_convtranspose1d_should_throw_type_error():
    with pytest.raises(TypeError) as ex:
        x = ConvTranspose1d()



@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size,input_shape, stride, padding, output_padding, groups, bias, dilation,name",
    [
        #Checking in_channels validation
        (0.3,0.3,0.36,"invalid","invalid","invalid","invalid","groups",False,"invalid",""),
        (0.3,0.3,0.36,"invalid","invalid","invalid","invalid","groups",False,"invalid",""),

        #Checking out_channels validation
        (3,3,0.36,"invalid","invalid","invalid","invalid","groups",False,"invalid",""),
        (3,False,0.36,"invalid","invalid","invalid","invalid","groups",False,"invalid",""),
        (3,"",0.36,"invalid","invalid","invalid","invalid","groups",False,"invalid",""),
        (3,("",),0.36,"invalid","invalid","invalid","invalid","groups",False,"invalid",""),

        #Checking kernel_size validation
        (3,(3,),"invalid","invalid","invalid","invalid","invalid","groups",False,"invalid",""),
        (3,(3,),False,5.7,"invalid","invalid","invalid","groups",False,"invalid",""),
        (3,(3,),(3,""),"invalid","invalid","invalid","invalid","groups",False,"invalid",""),
        (3,(3,),("",3),"invalid","invalid","invalid","invalid","groups",False,"invalid",""),

        #Checking input_shape validation
        (3,(3,),(3,3),"invalid","invalid","invalid","invalid","groups",False,"invalid",""),
        (3,(3,),(3,3),5.7,"invalid","invalid","invalid","groups",False,"invalid",""),
        (3,(3,),(3,3),False,"invalid","invalid","invalid","groups",False,"invalid",""),
        (3,(3,),(3,3),("",),"invalid","invalid","invalid","groups",False,"invalid",""),
        (3,(3,),(3,3),(5.7,),"invalid","invalid","invalid","groups",False,"invalid",""),

        #Checking stride validation
        (3,(3,),(3,3),(3,),"invalid","invalid","invalid","groups",False,"invalid",""),
        (3,(3,),(3,3),(3,),False,"invalid","invalid","groups",False,"invalid",""),
        (3,(3,),(3,3),(3,),5.7,"invalid","invalid","groups",False,"invalid",""),
        (3,(3,),(3,3),(3,),("",),"invalid","invalid","groups",False,"invalid",""),

        #Checking padding validation
        (3,(3,),(3,3),(3,),(3,3),"invalid","invalid","groups",False,"invalid",""),
        (3,(3,),(3,3),(3,),(3,3),False,"invalid","groups",False,"invalid",""),
        (3,(3,),(3,3),(3,),(3,3),5.7,"invalid","groups",False,"invalid",""),
        (3,(3,),(3,3),(3,),(3,3),("",),"invalid","groups",False,"invalid",""),

        #Checking output_padding validation
        (3,(3,),(3,3),(3,),(3,3),(3,),"invalid","groups",False,"invalid",""),
        (3,(3,),(3,3),(3,),(3,3),(3,),False,"groups",False,"invalid",""),
        (3,(3,),(3,3),(3,),(3,3),3,5.7,"groups",False,"invalid",""),
        (3,(3,),(3,3),(3,),(3,3),(3,3),("",),"groups",False,"invalid",""),

        #Checking groups validation
        (3,(3,),(3,3),(3,),(3,3),(3,),(3,3),"groups",False,"invalid",""),
        (3,(3,),(3,3),(3,),(3,3),(3,),(3,3),1,False,"invalid",""),
        (3,(3,),(3,3),(3,),(3,3),3,(3,3),5.7,False,"invalid",""),
        (3,(3,),(3,3),(3,),(3,3),(3,3),(3,3),("",),False,"invalid",""),

        #Checking bias validation
        (3,(3,),(3,3),(3,),(3,3),(3,),(3,3),3,"invalid","invalid",""),
        (3,(3,),(3,3),(3,),(3,3),(3,),(3,3),3,False,"invalid",""),
        (3,(3,),(3,3),(3,),(3,3),3,(3,3),3,3,"invalid",""),
        (3,(3,),(3,3),(3,),(3,3),(3,3),(3,3),3,5.7,"invalid",""),

        #Checking dilation validation
        (3,(3,),(3,3),(3,),(3,3),(3,),(3,3),3,False,"invalid",""),
        (3,(3,),(3,3),(3,),(3,3),(3,),(3,3),3,False,3,""),
        (3,(3,),(3,3),(3,),(3,3),3,(3,3),3,3,("",),""),
        (3,(3,),(3,3),(3,),(3,3),(3,3),(3,3),3,5.7,5.7,""),

        #Checking dilation validation
        (3,(3,),(3,3),(3,),(3,3),(3,),(3,3),3,False,(3,),"invalid"),
        (3,(3,),(3,3),(3,),(3,3),(3,),(3,3),3,False,(3,),False),
        (3,(3,),(3,3),(3,),(3,3),3,(3,3),3,3,(3,),3.5),
        (3,(3,),(3,3),(3,),(3,3),(3,3),(3,3),3,5.7,(3,),3),

    ]
)
def test_convtrans1d_should_throw_value_error(in_channels, out_channels, kernel_size, input_shape,stride,padding, output_padding,groups,bias,dilation,name):
    with pytest.raises(ValueError) as ex:
        x = ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, input_shape=input_shape, stride=stride, padding=padding, output_padding=output_padding,groups=groups,bias=bias,dilation=dilation, name=name)