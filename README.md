# **ICESEE**

Ice Sheet State and Parameter Estimator (ICESEE) model is a state-of-the-art data assimilation software package designed for ice sheet models. This advanced software facilitates the creation of an adaptive intelligent wrapper with robust protocols and APIs to seamlessly couple and integrate with various ice sheet models. The primary objective is to simplify the interaction between different models, enabling the adoption of complex data assimilation techniques across multiple frameworks.

This design is being extended to integrate with cloud computing services such as **AWS**, ensuring scalability and efficiency for larger simulations. Eventually, the software will be incorporated into the **GHUB online ice sheet platform**, significantly enhancing its capabilities by including the new features currently under development.

---

## **Usage**

The supported applications are located in the [applications](./applications) directory and currently include:
- **[Flowline](./applications/flowline/)**
- **[Icepack](./applications/icepack/)**
- **[Lorenz-96](./applications/lorenz-96/)**

### **Running Icepack in Containers**
Icepack applications can now be run in containers using both **Apptainer** and **Docker**, making them suitable for high-performance computing (HPC) clusters. For details, see [/src/container/apptainer](./src/container/apptainer/).

---

### **Running Applications with Data Assimilation**
Each application includes either a Python script or a Jupyter notebook for execution. Detailed documentation for these scripts and notebooks is included in the README files in each application folder. The documentation and full implementation of flowline model is forthcoming.

Both  **Icepack** and **Lorenz-96** applications support four variants of the Ensemble Kalman Filter for data assimilation:
1. **ENEnKF**: Stochastic Ensemble Kalman Filter
2. **DEnKF**: Deterministic Ensemble Kalman Filter
3. **EnTKF**: Ensemble Transform Kalman Filter
4. **EnRSKF**: Ensemble Square Root Kalman Filter

These variants enable robust and scalable data assimilation techniques tailored for ice sheet modeling.

---
