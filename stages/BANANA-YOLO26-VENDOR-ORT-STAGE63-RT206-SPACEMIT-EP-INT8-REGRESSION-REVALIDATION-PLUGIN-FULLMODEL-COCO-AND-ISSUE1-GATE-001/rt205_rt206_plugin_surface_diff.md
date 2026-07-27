# Plugin surface delta

The 2.0.6 archive adds a top-level `plugin/` tree, plugin build/run scripts,
test models, and missing sample documentation. The public
`spacemit_ort_plugin.h` header and existing `samples/plugin` implementation
sources are byte-identical to 2.0.5.

The material ABI change is in `libspacemit_ep.so.2`: 2.0.6 exports the public
`SpinePluginNode`, `SpinePluginTensor`, `SpinePluginContext`, and graph-viewer
methods that were unresolved in 2.0.5. Static and dynamic execution results are
reported separately because symbol resolution alone does not prove correct
custom-operator execution.
