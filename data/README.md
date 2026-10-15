# Datasets

Raw data is git-ignored. Put CSV files here and point `data.csv_path` at them.

Public IIoT intrusion datasets that fit the CSV loader:

- Edge-IIoTset (Ferrag et al., 2022). Label column `Attack_type`, benign label `Normal`.
- X-IIoTID (Al-Hawawreh et al., 2021). Label column `class1`, benign label `Normal`.
- TON_IoT network data (Moustafa, 2021). Label column `type`, benign label `normal`.

Identifier columns such as IPs, ports, timestamps and raw payloads should be
listed under `data.drop_columns` so the model doesn't learn shortcuts from them.
For Edge-IIoTset that also means dropping `Attack_label`, which leaks the target.
