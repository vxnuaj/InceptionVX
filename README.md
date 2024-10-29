<div align = 'center'>
<img src = 'https://miro.medium.com/v2/resize:fit:2000/1*UItPkoIvPZR5iXgzVgap6g.png'>
</div>

## InceptionV1

Implementation of InceptionV1, proposed on [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) by Szegedy et al

### Index

1. [InceptionV1](InceptionV1)
   1. [Implementation](InceptionV1/inceptionv1.py)
   2. [Notes](InceptionV1/V1notes.md)
2. [InceptionV3](InceptionV3)
   1. [Implementation](InceptionV3/inceptionv3.py)
   2. [Notes](InceptionV3/V3notes.md)

### Usage

1. Clone the Repo
2. Run `inceptionv1_run.py`

    ```python
    import torch
    from torchinfo import summary
    from inceptionv1 import InceptionV1

    # init random tensor
    x = torch.randn(1, 3, 224, 224)

    # init model
    model = InceptionV1()

    # model summary
    summary(model, input_data = x)
    ```

## InceptionV3

Implementation of InceptionV3, proposed on [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) by Szegedy et al

### Usage

1. Clone the Repo
2. Run `inceptionv3_run.py`

    ```python
    import torch
    from torchinfo import summary
    from inceptionv1 import InceptionV1

    # init random tensor
    x = torch.randn(1, 3, 224, 224)

    # init model
    model = InceptionV1()

    # model summary
    summary(model, input_data = x)
    ```