# pamodpy
The Power-in-the-Loop Autonomous Mobility-on-Demand (P-AMoD) Python Toolkit optimizes for planning and operational decisions at lowest cost for autonomous electric ride-hail fleet managers.
The Planning Tool determines fleet sizing with consideration of mixed vehicle models and powertrains, charging station siting with consideration of various station types and charging rates, and fleet operations (i.e., rebalancing, charging, serving customers, and idling) at a mesoscopic level.
The Operations Tool determines fleet operations at a microscopic level with real-time performance speeds.

pamodpy is licensed under the MIT License (see LICENSE file). The copyright notice and permission notice in LICENSE shall be included in all
copies or substantial portions of pamodpy.

## Usage
1) Build the Anaconda environment using environment_from_history.yml
2) Obtain a Gurobi license on your machine
3) Configure your experiment by creating a .json file in experiment_configs
4) Run the toolkit by executing main.py 
5) Results are stored in the results directory

### Obtaining a Gurobi Academic License on a Virtual Machine
1) Ensure Gurobi is installed on your virtual machine (Gurobi is one of the Anaconda packages in environment_from_history.yml). Check the Gurobi version number installed. 
2) On Gurobi's website, register for an Academic License. Note your license key.
3) On the virtual machine command line, run grbprobe and note the hostname, hostid, username, and platform
4) On your computer that is connected to an academic network (e.g., campus internet, school VPN), make the following HTTP GET request: https://apps.gurobi.com/keyserver?id=<key>&hostname=<hostname>&hostid=<hostid>&username=<username>&os=<platform>&localdate=<YYYY-MM-DD>&version=<version>
5) Save the returned information as a gurobi.lic file
* The above instructions were adapted from this article: https://sproul.xyz/blog/posts/gurobi-academic-validation.html

## References
If you use pamod-py in an academic context, please acknowledge this and cite the following article:
J. Luke, M. Salazar, R. Rajagopal and M. Pavone, "Joint Optimization of Autonomous Electric Vehicle Fleet Operations and Charging Station Siting," 2021 IEEE International Intelligent Transportation Systems Conference (ITSC), 2021, pp. 3340-3347, doi: 10.1109/ITSC48978.2021.9565089.
https://arxiv.org/abs/2107.00165

## Support
* Contact Justin Luke at [firstname].[lastname]@stanford.edu
