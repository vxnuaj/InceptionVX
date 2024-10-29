<div align = 'center'>
<img src = 'https://miro.medium.com/v2/resize:fit:2000/1*UItPkoIvPZR5iXgzVgap6g.png'>
</div>

## InceptionV1

Implementation of InceptionV1, proposed on [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) by Szegedy et al

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