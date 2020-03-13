# Ben's Repository for the L7 Computer Vision Group Project

## Data collection

Data collection is split into 3 parts:
- Simple
- Moderate
- Hard

These are outlined in the [Project porposal](https://docs.google.com/document/d/1W9PfvrghWIqlU7HPn4r41bp9XF29-lf22zSJjPwU78k/edit) but are summarised below.

### Simple collection
Extract a dataset by driving around in GTAV using:
- one environment
- constant weather conditions
- no occlusions
- no data augmentation

This will generate a first batch so initial traditional computer vision models have data to work with.

### Moderate collection
- Use a bot to automatically drive and collect data
- Store data in a more organised fashion with better naming conventions and foldering
- Use GTA mods to alter the weather conditions
- Change times of day
- New locations such as city and off-road
- Low-level data augmentation such as translations and reflections

This will create a larger dataset ready to be used for Deep Learning models

### Hard collection
- Increase dataset size with more autonomous collection
- Better data augmentation
- Add in varied occlusions and domain adaptation
- Image style transfer from real-world data to synthetic data
