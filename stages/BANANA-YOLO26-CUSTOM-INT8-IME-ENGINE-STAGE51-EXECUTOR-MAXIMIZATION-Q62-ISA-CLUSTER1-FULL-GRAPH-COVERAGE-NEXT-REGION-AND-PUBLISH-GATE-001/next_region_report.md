# Next resident region report

Region F is exact for Python/portable C++ host scalar and board scalar/IME across F0-F7 at all 39
package boundaries. The selected runtime sidecar uses CPU4 only as controller; IME remains on
CPU0-3. The region has zero internal conversions and zero float materializations.

- custom region: 3455.120892 us mean / 3514.509150 us p95.
- matched B120 ORT region: 19115.925104 us mean / 19123.943940 us p95.
- delta: -81.925432% mean / -81.622467% p95.
- selected combined model4-final to model9: 20973.341840 us mean / 21161.386150 us p95.
- matched B120 ORT combined: 77055.672360 us mean / 77303.342360 us p95.

Classification: strong-positive. This proves one more resident region, not a full-model executor.
