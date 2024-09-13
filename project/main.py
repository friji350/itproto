import mujoco
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time


model = mujoco.MjModel.from_xml_path('mjmodel.xml')
data = mujoco.MjData(model)

configurations = []

for step in range(3 ** model.nq):
    indices = np.unravel_index(step, (3,) * model.nq)
    configuration = [np.linspace(model.jnt_range[j][0], model.jnt_range[j][1], 3)[indices[j]] for j in
                         range(model.nq)]
    configurations.append(configuration)

a = []
t = []

for config in configurations:
    data.qpos[:] = config
    mujoco.mj_inverse(model, data)
    a.append(data.qpos.copy())
    t.append(data.qfrc_inverse.copy())

a_df = pd.DataFrame(a, columns=[f'Joint_{i + 1}' for i in range(model.nq)])
t_df = pd.DataFrame(t, columns=[f'Torque_{i + 1}' for i in range(model.nq)])

df = pd.concat([a_df, t_df], axis=1)
df.to_csv('res.csv', index=False)

plt.figure(figsize=(12, 6))

data = t_df.map(lambda x: np.log(np.abs(x) + 1e-6)).reset_index()

data = pd.melt(data, id_vars='index', var_name='Joint', value_name='Torque')
sns.violinplot(x='Joint', y='Torque', data=data)
plt.xlabel('Joint')
plt.ylabel('Torque')
plt.show()

plt.savefig('torque_distribution.png')
plt.close()
