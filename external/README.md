# External Source Trees

This directory is for local clones used during reading and adaptation.

The actual third-party source trees are ignored by this repository to avoid vendoring nested git repositories:

```bash
git clone https://github.com/Koopman-Laboratory/KoopmanLab external/KoopmanLab
git clone https://github.com/Quin557/am_fno_repro external/am_fno_repro
```

Use submodules later only if the experiment repository must pin exact upstream commits.
