# MTCNN

## Instructions to Train and Test the Model:

```shell
mkdir prepare_data/Data
mkdir prepare_data/Combined_Data
```

* Training:

```shell
python wider_face_bbox_generate.py
```

```shell
python combined_data_pnet.py
```

```shell
python training/train_p_net.py
```

```shell
python rnet_generate.py
```

```shell
python combined_data_rnet.py
```

```shell
python training/train_r_net.py
```

```shell
python onet_generate.py
```

```shell
python combined_data_onet.py
```

```shell
python training/train_o_net.py
```
* Testing:

```shell
python demo/test.py
```
