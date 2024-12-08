//模型运行流程
1.首先对一部分数据集进行打标签（转换成yolo框标签/编辑对应的classes.txt文件/采用images、labels、txt在一个my-XX的文件夹下）
2.划分数据集，将myseg，对应地址修改，放入同一文件目录下train_XX，用于划分数据集和测试集，运行seg.py.
3.将分好的数据集，放入\datasets\XX dataset 里
4.修改配置文件：模型配置文件cfg\models\v8\yaml,根据自己的数据集更改分类数
数据集配置文件：cfg\datasets\yaml,根据数据集更改位置
5.训练模型：最外层demo.py，运行demo.


//添加模块，nn/modules/conv.py and init.py里是否有，以CBAM为例，然后tasks.py查看，没有就添加上去
修改配置文件,修改yaml文件，XXyolov8_CBAM.yaml,(问题，注意力机制模块应该加在什么位置)
修改ultralytics\nn\tasks.py，在parse_model加上CBAM，看809行代码或搜索CBAM


//添加自定义模块：以自定义的CBAM为例
ultralytics\nn\modules\conv.py添加myCBAM类，类似于自带的CBAM，代码如zizhi\myCBAMmodel.py所见
然后ultralytics\nn\modules\conv.py的__all__ = 中添加my_CBAM，
ultralytics\nn\modules\__init__.py的from .covn import 中添加my_CBAM，__all__ = 中添加my_CBAM
在ultralytics\nn\tasks.py中from ultralytics.nn.modules import 和parse_model的elif添加my_CBAM，与自带的CBAM类似，可直接改成elif m in (CBAM, my_CBAM):
接着修改配置的yaml文件即可