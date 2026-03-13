# Project-Good-Dataset-PMU-Data-Reconstruction
Real-Time Synchrophasor Dataset Enabling AI for Secure and Resilient Power Grids
<img width="1180" height="428" alt="image" src="https://github.com/user-attachments/assets/60266d63-ba3a-408f-ae07-eaaab1d378e1" />

This dataset contains real-time, synchronized synchrophasor measurements acquired from industrial Phasor Measurement Units (PMUs) for power system monitoring and data-driven analysis. The voltage and frequency measurements are transmitted by a hardware PMU (SEL-421-4, functioning as a protection relay) and a Real-Time Digital Simulator (RTDS)-based GTNETx2 PMU at 50 frames/second to the Real-Time Automation Controller (SEL-RTAC). The on-board Dynamic Disturbance Recorder (DDR) facility on the SEL-3530 RTAC was used for continuous periodic sampling (one sample every 20 ms) of streaming frequency measurements. The dataset recoded provides precise timestamps with every frame, capturing cyberphysical data integrity conditions, including (1) noisy, (2) fault resembling attack, (3) data repetition attack, (4) Man-In-the-Middle (MITM) or data manipulation attack and (5) data missing attack. These measurements accurately reflect synchronization characteristics, communication impairments, and operational signal fluctuations in power systems. Datasets are organized in structured .csv files with clearly defined metadata that describe measurement sources and operating scenarios. These datasets are intended for artificial intelligence- and data-driven-based applications: anomaly detection, data reconstruction, resilient power grid monitoring, and wide-area controls in smart grids.

Four types of streaming PMU data attack scenario:\\
    Case I: Man-in-the-Middle (MITM)\\
    Case II: Topology Resembling Attack\\
    Case III: Data Repetition Attack\\
    Case IV: Data Missing Attack \\
