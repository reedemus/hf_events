# Background

a Stable Diffusion(SD) demo web-app developed as part of a [Huggingface Community event](https://huggingface.co/keras-dreambooth). The web app generates images of Harry Potter characters with **new** style of outfits inspired by [_Hogwarts Legacy's_](https://www.hogwartslegacy.com/) video game, given a unique identifier token **hogwarts [legacy] student** in the prompt's sentence. The custom SD model is fine-tuned using Google's [Dreambooth](https://arxiv.org/abs/2208.12242) method.

# What is Dreambooth?
[Dreambooth](https://arxiv.org/abs/2208.12242) is a technique developed by Google's research team in 2022 to fine-tune diffusion models(like Stable Diffusion) by injecting a custom subject to the model. This enables any ** _customized subject_ ** (i.e. your pet dog) appear in different scenes, poses, and views in the generated image. To read more about it, visit the project's blog at [https://dreambooth.github.io](https://dreambooth.github.io/).

Try the app here at [HF Spaces DreamBooth Hogwarts Legacy!](https://huggingface.co/spaces/keras-dreambooth/dreambooth_hogwarts_legacy)

![example.png](https://i.postimg.cc/y8zq29kw/example.png)