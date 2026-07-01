# HarmGen: Improving Evaluation of Heterogeneous Congestion Control Algorithm Interactions

## Code Structure
```
├── harm-exp
│   ├── draw_distribution_long_flows.py         # plot codes
│   ├── draw_distribution_short_flows.py
│   ├── draw_heatmap.py
│   ├── draw_training.py
│   ├── extract_harm_long_flows.py              # extract results to csv files
│   ├── extract_harm_short_flows.py      
│   ├── genetic-algorithm.py                    # core code of HarmGen
│   ├── mahak_helper.py
│   ├── mahak.py                                # core code of Mahak
│   ├── mahimahi_single_run.py                  # experiment setup using mahimahi
│   ├── modAL                                   # Mahak's AL model
│   ├── run_harmgen_long_flow.sh                # scripts to run codes
│   ├── run_harmgen_short_flow.sh
│   ├── run_mahak.sh
│   └── tc_single_run.py                        # experiment setup using tc
├── README.md
└── Vagrantfile                                 # VM setup
```

## VM setup
Using following code to setup vagrant and libvirt
``` bash
echo 'Installing vagrant and libvirt...'
curl -O https://raw.githubusercontent.com/vagrant-libvirt/vagrant-libvirt-qa/main/scripts/install.bash
chmod a+x ./install.bash
./install.bash || exit 1
rm ./install.bash

echo 'Installing vagrant plugins...'
vagrant plugin install vagrant-rsync-back

echo 'Enable the ports used by nfs...'
# 192.168.121.x
if command -v ufw >/dev/null 2>&1; then
	sudo ufw allow from 192.168.121.0/24 || echo "Fail to set ufw rule..."
else
	echo "Please do remember to allow the connections from the private network of VM in your firewall."
fi

echo 'Grant the user privilege... '
sudo usermod -aG kvm "$USER"
sudo usermod -aG libvirt "$USER"
```

Some of Vagrant commands:
* `vagrant up`: Start VM
* `vagrant ssh`: Connect to VM
* `vagrant rsync`: Sync host files to VM
* `vagrant rsync-back`: Sync VM files to host
* `vagrant reload`: Restart VM
* `vagrant halt`: shutdown VM
* `vagrant destroy -f`: Destroy VM

Destroying VM manually:
``` bash
# find the name of the VM
sudo virsh list --all
sudo virsh shutdown <vm-name>
sudo virsh destroy <vm-name>
sudo virsh undefine <vm-name>
```