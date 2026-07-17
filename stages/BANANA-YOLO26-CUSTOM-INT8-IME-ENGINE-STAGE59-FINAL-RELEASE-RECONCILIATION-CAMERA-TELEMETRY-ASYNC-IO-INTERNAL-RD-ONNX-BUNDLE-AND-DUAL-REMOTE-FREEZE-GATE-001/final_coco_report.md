# Final COCO Validation

The final 0.9.2 source completed COCO val2017 5000/5000 under the selected O2
profile. It emitted 721,755 predictions and an 81,422,145-byte JSON whose
SHA-256 is byte-identical to the accepted Stage58 prediction:

`cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda`

Byte identity preserves all accepted metrics, including mAP50-95
`0.3707408944391919`, mAP50 `0.5258465300872381`, AP small
`0.18397294626227842`, AP medium `0.4142627352606523`, and AP large
`0.5440433811804918`. The package manifest was the expected
`fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be`.
The one-pass executor mean was 142661.912836 us and complete image/prediction
wall time was 850928599.298 us. O2 state was restored after the run.

The raw JSON remains on board NVMe under the Stage59 root; its hash, timing TSV,
thermal trace, and command output are retained in raw evidence.
