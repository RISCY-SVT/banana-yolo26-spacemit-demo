# RT205 plugin mechanism

The release implements mechanism C: a SpacemiT-EP-specific custom operator
plugin API (`spacemit_ort_plugin.h`, `SpacemitPluginInit`, provider option
`SPACEMIT_EP_PLUGIN_LIB`). It is distinct from standard ORT custom domains and
generic plugin EP loading.

The official sample and independent exact uint8 plugin both compile. Neither can
load against the shipped package: public API methods declared in the header,
including `SpinePluginTensor::GetDataType()`, are unresolved, and the package EP
does not export them. The independent session fails at `dlopen` before operator
registration or partitioning. The mechanism is package-present but ABI-broken.
