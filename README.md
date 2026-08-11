# Improving Evaluation of Heterogeneous Congestion Control Algorithm Interactions (ACM SIGCOMM 2026)

A new framework for evaluating fairness between heterogeneous congestion control algorithms (CCAs). We integrate it into **Mahak**, an existing active-learning tool, and present **HarmGen**, a new genetic algorithm that searches for the network settings where two CCAs interact worst.

This artifact reproduces the paper's Section 7 case studies: the evolution of BBR (Cubic vs BBRv1 against Cubic vs BBRv3), and L4S (TCP Prague vs Cubic over a DualPI2 AQM).

> Section 6 runs its experiments on [BESS](https://github.com/NetSys/bess), a software switch whose environment takes considerable effort to build and is hard to stand up quickly. Section 7 instead uses emulation tools that any Linux machine already has — tc and mahimahi — which is why we present artifact that reproduces Section 7. Note that swapping BESS for these emulation tools only changes how a single experiment's result is measured; the genetic algorithm of HarmGen and the active-learning framework of Mahak, that is, the core code, are unchanged.

The experiments run in two Vagrant/libvirt VMs. BBRv3 and DualPI2 + TCP Prague patch the same kernel subsystems and cannot coexist in one tree, so each gets its own VM with its own copy of `harm-exp`.

| Folder | VM name | Static IP | Kernel |
| --- | --- | --- | --- |
| `vm-bbrv3` | `vm-harmgen-bbrv3` | 192.168.121.5 | 6.13.7 + BBRv3 |
| `vm-dualpi2-prague` | `vm-harmgen-dualpi2-prague` | 192.168.121.15 | 5.15.72 + DualPI2 + TCP Prague |

Each folder is a standalone Vagrant environment: run all `vagrant` commands from inside it. There is no Vagrantfile at the repo root.

## Table of Contents

- [Code Structure](#code-structure)
- [Setting up the VMs](#setting-up-the-vms)
  - [Host prerequisites](#host-prerequisites)
  - [VM 1 — BBRv3](#vm-1--bbrv3)
    - [1. Create the SSH key](#1-create-the-ssh-key)
    - [2. Start the VM and install the key](#2-start-the-vm-and-install-the-key)
    - [3. Get the sources and patch the config](#3-get-the-sources-and-patch-the-config)
    - [4. Build and install](#4-build-and-install)
    - [5. Verify](#5-verify)
  - [VM 2 — DualPI2 + TCP Prague](#vm-2--dualpi2--tcp-prague)
    - [1. Create the SSH key](#1-create-the-ssh-key-1)
    - [2. Start the VM and install the key](#2-start-the-vm-and-install-the-key-1)
    - [3. Get the sources at the pinned commit](#3-get-the-sources-at-the-pinned-commit)
    - [4. Configure](#4-configure)
    - [5. Build and install](#5-build-and-install)
    - [6. Verify](#6-verify)
    - [7. Build the patched iproute2](#7-build-the-patched-iproute2)
  - [Common Vagrant commands](#common-vagrant-commands)
  - [Troubleshooting](#troubleshooting)
  - [Removing a VM](#removing-a-vm)
- [Reproducing Section 7](#reproducing-section-7)
  - [Experiment dependencies](#experiment-dependencies)
  - [Per-boot configuration](#per-boot-configuration)
  - [Section 7.1 — Cubic vs BBRv3](#section-71--cubic-vs-bbrv3)
  - [Section 7.2 — L4S: TCP Prague vs Cubic](#section-72--l4s-tcp-prague-vs-cubic)
  - [Runtime and retrieving results](#runtime-and-retrieving-results)
- [Where to find the results](#where-to-find-the-results)
  - [Section 7.1 — Mahak results](#section-71--mahak-results)
  - [Section 7.2 — HarmGen results](#section-72--harmgen-results)

## Code Structure

The two `harm-exp` trees hold identical `.py` files; only the run scripts differ, because each VM can only exercise the CCAs its own kernel provides.

```
├── vm-bbrv3                                    # VM 1: BBRv3
│   ├── Vagrantfile
│   └── harm-exp                                # synced to /home/vagrant/harm-exp
│       ├── genetic-algorithm.py                # core code of HarmGen
│       ├── mahak.py                            # core code of Mahak
│       ├── mahak_helper.py
│       ├── modAL                               # Mahak's AL model
│       ├── tc_single_run.py                    # experiment setup using tc, harm + PACE convergence
│       ├── mahimahi_single_run.py              # experiment setup using mahimahi 
│       ├── extract_harm_long_flows.py          # extract results to csv files
│       ├── extract_harm_short_flows.py
│       ├── draw_distribution_long_flows.py     # plot codes
│       ├── draw_distribution_short_flows.py
│       ├── draw_heatmap.py                     # Mahak's predicted-harm heatmaps
│       ├── draw_training.py                    # Mahak's sampling heatmaps
│       ├── plot_top_harm.py                    # highest-harm settings found by HarmGen
│       ├── plot_mahak_compare.py               # BBRv1 vs BBRv3 across Mahak's predictions
│       ├── run_mahak.sh                        # scripts to run codes
│       ├── run_harmgen_long_flow.sh
│       ├── run_harmgen_short_flow.sh
│       └── run_section7_1.sh                   # full Section 7.1 reproduction
├── vm-dualpi2-prague                           # VM 2: DualPI2 + TCP Prague
│   ├── Vagrantfile
│   └── harm-exp                                # same .py files; run_section7_2.sh instead of 7_1
└── README.md
```

Results are written into `harm-exp/mahak_results/` (Mahak) and `harm-exp/results/` (HarmGen). Both are gitignored and created on the first run.

## Setting up the VMs

### Host prerequisites

Vagrant + libvirt:

``` bash
curl -O https://raw.githubusercontent.com/vagrant-libvirt/vagrant-libvirt-qa/main/scripts/install.bash
chmod a+x ./install.bash
./install.bash || exit 1
rm ./install.bash

vagrant plugin install vagrant-rsync-back

# Allow the VM private network through the firewall
if command -v ufw >/dev/null 2>&1; then
	sudo ufw allow from 192.168.121.0/24 || echo "Fail to set ufw rule..."
fi

sudo usermod -aG kvm "$USER"
sudo usermod -aG libvirt "$USER"
```

Log out and back in, then check:

``` bash
id -nG                                 # must list kvm and libvirt
virsh -c qemu:///system list --all     # must work without sudo
```

Kernel build tools (both kernels are compiled **on the host**, not in the VMs):

``` bash
sudo apt update
sudo apt install -y git build-essential bc bison flex libssl-dev libelf-dev \
                    libncurses-dev dwarves rsync cpio kmod dpkg-dev debhelper fakeroot
```

Requires ~15 GB free disk. On 24 cores each kernel takes 10–25 minutes.

### VM 1 — BBRv3

#### 1. Create the SSH key

`gce-install.sh` calls bare `ssh`/`scp` with no `-i` flag, so the key must be bound in `~/.ssh/config`.

``` bash
ssh-keygen -t ed25519 -f ~/.ssh/harmgen_bbrv3_ed25519 -N "" -C "harmgen-bbrv3-vm"

cat >> ~/.ssh/config <<'EOF'

# --- HarmGen BBRv3 VM ---
Host 192.168.121.5 vm-harmgen-bbrv3
  HostName 192.168.121.5
  User vagrant
  IdentityFile ~/.ssh/harmgen_bbrv3_ed25519
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
  UserKnownHostsFile ~/.ssh/known_hosts_harmgen
# --- end HarmGen BBRv3 VM ---
EOF
chmod 600 ~/.ssh/config
```

#### 2. Start the VM and install the key

``` bash
cd vm-bbrv3
vagrant up --provider=libvirt

PUBKEY="$(cat ~/.ssh/harmgen_bbrv3_ed25519.pub)"
vagrant ssh -c "
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  grep -qxF '$PUBKEY' ~/.ssh/authorized_keys 2>/dev/null || echo '$PUBKEY' >> ~/.ssh/authorized_keys
  chmod 600 ~/.ssh/authorized_keys
"
```

Do not continue until this works:

``` bash
ssh vagrant@192.168.121.5 "hostname; sudo -n true && echo sudo-ok"
```

#### 3. Get the sources and patch the config

``` bash
cd ~
git clone -o google-bbr -b v3 --depth 1 https://github.com/google/bbr.git
cd ~/bbr
git config core.abbrev 12        # makes the resulting uname -r reproducible
```

`config.gce` targets Google Compute Engine and ships with `CONFIG_VIRTIO_BLK` and `CONFIG_VIRTIO_BALLOON` unset. The VM's root disk is `vda` (virtio-blk), so **without `CONFIG_VIRTIO_BLK=y` the VM will not boot.**

``` bash
cp config.gce config.gce.orig
sed -i 's/^# CONFIG_VIRTIO_BLK is not set$/CONFIG_VIRTIO_BLK=y/
        s/^# CONFIG_VIRTIO_BALLOON is not set$/CONFIG_VIRTIO_BALLOON=y/' config.gce

for f in CONFIG_VIRTIO=y CONFIG_VIRTIO_NET=y CONFIG_VIRTIO_PCI=y \
         CONFIG_VIRTIO_BALLOON=y CONFIG_VIRTIO_BLK=y CONFIG_NET=y \
         CONFIG_NETDEVICES=y CONFIG_ETHERNET=y; do
  grep -qx "$f" config.gce && echo "OK   $f" || echo "MISS $f"
done
```

All eight must print `OK`.

#### 4. Build and install

``` bash
cd ~/bbr
./gce-install.sh -m vagrant@192.168.121.5
```

The script builds on the host, copies the kernel to the VM, and reboots it. It ends with `Connection ... closed by remote host` and **exit code 255** — expected, because its last remote command is `sudo reboot`. Compiler output is in `/tmp/make.*`.

Wait for the VM to come back:

``` bash
until ssh -o ConnectTimeout=5 vagrant@192.168.121.5 "uname -r" 2>/dev/null | grep -q GCE; do sleep 5; done
```

#### 5. Verify

``` bash
ssh vagrant@192.168.121.5 "uname -r; sudo modinfo tcp_bbr | head -3; sysctl net.ipv4.tcp_available_congestion_control"
```

```
6.13.7+v3+90210de4b779+GCE
name:           tcp_bbr
filename:       (builtin)
version:        3
net.ipv4.tcp_available_congestion_control = reno bbr bbr1 bic cdg cubic dctcp westwood highspeed hybla htcp vegas nv veno scalable lp yeah illinois
```

`version: 3` confirms BBRv3. `bbr` is v3, `bbr1` is v1.

### VM 2 — DualPI2 + TCP Prague

Source: <https://github.com/L4STeam/linux>, pinned to commit `48b3db6b4a7fd57e2d31db3bb46a3bc6af7bf3ad` = **Linux 5.15.72**.

> This part of experiments of this paper were done on kernel version on 5.15.72.

#### 1. Create the SSH key

``` bash
ssh-keygen -t ed25519 -f ~/.ssh/harmgen_dualpi2_ed25519 -N "" -C "harmgen-dualpi2-prague-vm"

cat >> ~/.ssh/config <<'EOF'

# --- HarmGen DualPI2+Prague VM ---
Host 192.168.121.15 vm-harmgen-dualpi2-prague
  HostName 192.168.121.15
  User vagrant
  IdentityFile ~/.ssh/harmgen_dualpi2_ed25519
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
  UserKnownHostsFile ~/.ssh/known_hosts_harmgen
# --- end HarmGen DualPI2+Prague VM ---
EOF
chmod 600 ~/.ssh/config
```

#### 2. Start the VM and install the key

``` bash
cd vm-dualpi2-prague
vagrant up --provider=libvirt

PUBKEY="$(cat ~/.ssh/harmgen_dualpi2_ed25519.pub)"
vagrant ssh -c "
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  grep -qxF '$PUBKEY' ~/.ssh/authorized_keys 2>/dev/null || echo '$PUBKEY' >> ~/.ssh/authorized_keys
  chmod 600 ~/.ssh/authorized_keys
"

ssh vagrant@192.168.121.15 "hostname; sudo -n true && echo sudo-ok"
```

#### 3. Get the sources at the pinned commit

``` bash
mkdir -p ~/l4s-linux && cd ~/l4s-linux
git init
git remote add origin https://github.com/L4STeam/linux.git
git fetch --depth 1 origin 48b3db6b4a7fd57e2d31db3bb46a3bc6af7bf3ad
git checkout FETCH_HEAD
make kernelversion        # must print 5.15.72
```

#### 4. Configure

Base the config on the VM's own running kernel:

``` bash
cd ~/l4s-linux
scp vagrant@192.168.121.15:/boot/config-5.15.0-91-generic ./.config

# Ubuntu's config points at Canonical certs that are absent here; the build
# fails without clearing them
scripts/config --set-str SYSTEM_TRUSTED_KEYS ""
scripts/config --set-str SYSTEM_REVOCATION_KEYS ""

# Debug info makes the build much slower and the .deb enormous
scripts/config --disable DEBUG_INFO
scripts/config --disable DEBUG_INFO_BTF
scripts/config --disable DEBUG_INFO_DWARF4

# L4S modules
scripts/config -m TCP_CONG_PRAGUE
scripts/config -m NET_SCH_DUALPI2
scripts/config -m TCP_CONG_DCTCP
scripts/config -m TCP_CONG_BBR2

make olddefconfig
grep -E "^(CONFIG_TCP_CONG_PRAGUE|CONFIG_NET_SCH_DUALPI2|CONFIG_TCP_CONG_DCTCP|CONFIG_TCP_CONG_BBR2)=" .config
```

All four must print `=m`.

#### 5. Build and install

``` bash
cd ~/l4s-linux
make -j$(nproc) bindeb-pkg LOCALVERSION=-prague-1 KDEB_PKGVERSION=1

# Copy the packages to the VM and install
scp ../linux-image-5.15.72-prague-1_1_amd64.deb \
    ../linux-headers-5.15.72-prague-1_1_amd64.deb vagrant@192.168.121.15:~/
ssh vagrant@192.168.121.15 "sudo dpkg --install ~/linux-*-prague-1_1_amd64.deb && sudo update-grub && sudo reboot"
```

Wait for the VM to come back:

``` bash
until ssh -o ConnectTimeout=5 vagrant@192.168.121.15 "uname -r" 2>/dev/null | grep -q prague; do sleep 5; done
```

#### 6. Verify

``` bash
ssh vagrant@192.168.121.15 "
  uname -r
  sudo modprobe sch_dualpi2 tcp_prague
  lsmod | grep -E 'dualpi2|prague'
  sysctl net.ipv4.tcp_available_congestion_control
"
```

```
5.15.72-prague-1
tcp_prague             24576  0
sch_dualpi2            24576  0
net.ipv4.tcp_available_congestion_control = reno cubic prague
```

`prague` appears only after `modprobe tcp_prague`. `dctcp` and `bbr2` are built as modules too — `modprobe` them if a run needs them.

#### 7. Build the patched iproute2

The stock `tc` cannot read or set DualPI2 parameters — it reports `[Unknown qdisc, optlen=104]`. Since the experiment scripts drive `tc`, build L4S's patched version inside the VM:

``` bash
ssh vagrant@192.168.121.15 "
  sudo apt-get update
  sudo apt-get install -y gcc make bison flex pkg-config libdb-dev libmnl-dev libelf-dev libbpf-dev
  cd ~ && git clone https://github.com/L4STeam/iproute2.git
  cd iproute2 && ./configure && make -j\$(nproc)
"
```

This leaves the patched binary at `~/iproute2/tc/tc` (iproute2-5.12.0) without overwriting the system `tc`. Use it for all DualPI2 configuration:

``` bash
sudo ~/iproute2/tc/tc qdisc replace dev eth1 root dualpi2 \
     target 15ms tupdate 16ms alpha 0.16 beta 3.2
sudo ~/iproute2/tc/tc qdisc show dev eth1
```

```
qdisc dualpi2 8001: root refcnt 2 limit 10000p target 15ms tupdate 16ms alpha 0.152344 beta 3.195312 l4s_ect coupling_factor 2 drop_on_overload step_thresh 1ms drop_dequeue split_gso classic_protection 10%
```

For comparison, the stock `tc` on the same qdisc prints only:

```
qdisc dualpi2 8001: root refcnt 2 [Unknown qdisc, optlen=104]
```

### Common Vagrant commands

Run from inside `vm-bbrv3/` or `vm-dualpi2-prague/`:

| Command | Effect |
| --- | --- |
| `vagrant up` | Start VM |
| `vagrant ssh` | Connect to VM |
| `vagrant rsync` | Sync host files to VM |
| `vagrant rsync-back` | Sync VM files back to host |
| `vagrant reload` | Restart VM |
| `vagrant halt` | Shut down VM |
| `vagrant destroy -f` | Destroy VM |

> **Retrieve results before syncing.** Both rsync directions use `--delete`. `vagrant rsync` (host → VM) deletes VM-side files that are absent on the host, including experiment results. Always run `vagrant rsync-back` first.

### Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Permission denied (publickey)` | The key was not installed in the VM, or the `~/.ssh/config` block is missing. Check with `ssh -v`. |
| VM never returns after reboot | For `vm-bbrv3`, almost always `CONFIG_VIRTIO_BLK`. Inspect with `virsh -c qemu:///system console <domain>`; recover with `vagrant destroy -f && vagrant up`. |
| `uname -r` unchanged after install | The build failed but the script continued. Check `/tmp/make.*` (BBRv3) or the `make` output (L4S). |
| Results missing after `vagrant rsync` | `--delete` removed them. Use `vagrant rsync-back` first. |

### Removing a VM

Destroys the VM and deletes the SSH key, the `~/.ssh/config` block, and the dedicated `known_hosts` file. Save as `cleanup-vm.sh`, then `chmod +x cleanup-vm.sh && ./cleanup-vm.sh bbrv3` (or `dualpi2`):

``` bash
#!/usr/bin/env bash
# Usage: ./cleanup-vm.sh {bbrv3|dualpi2}
set -u

case "${1:-}" in
    bbrv3)
        VM_DIR=vm-bbrv3;            DOMAIN=vm-bbrv3_vm-harmgen-bbrv3
        KEY=~/.ssh/harmgen_bbrv3_ed25519;   MARKER='HarmGen BBRv3 VM'
        SRC=~/bbr ;;
    dualpi2)
        VM_DIR=vm-dualpi2-prague;   DOMAIN=vm-dualpi2-prague_vm-harmgen-dualpi2-prague
        KEY=~/.ssh/harmgen_dualpi2_ed25519; MARKER='HarmGen DualPI2+Prague VM'
        SRC=~/l4s-linux ;;
    *)  echo "Usage: $0 {bbrv3|dualpi2}"; exit 1 ;;
esac

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Destroying ${VM_DIR}..."
if [ -d "${REPO_DIR}/${VM_DIR}" ]; then
    ( cd "${REPO_DIR}/${VM_DIR}" && vagrant destroy -f ) || true
    rm -rf "${REPO_DIR}/${VM_DIR}/.vagrant"
fi

echo '==> Removing any leftover libvirt domain...'
if virsh -c qemu:///system dominfo "${DOMAIN}" >/dev/null 2>&1; then
    virsh -c qemu:///system destroy  "${DOMAIN}" 2>/dev/null || true
    virsh -c qemu:///system undefine "${DOMAIN}" --remove-all-storage 2>/dev/null || true
else
    echo '    none'
fi

echo '==> Removing the SSH key and config block...'
rm -fv "${KEY}" "${KEY}.pub"
if grep -q "^# --- ${MARKER} ---$" ~/.ssh/config 2>/dev/null; then
    cp ~/.ssh/config ~/.ssh/config.bak
    sed -i "/^# --- ${MARKER} ---$/,/^# --- end ${MARKER} ---$/d" ~/.ssh/config
    echo "    removed (backup at ~/.ssh/config.bak)"
fi

# Remove the dedicated known_hosts only once both VMs are gone
if [ ! -f ~/.ssh/harmgen_bbrv3_ed25519 ] && [ ! -f ~/.ssh/harmgen_dualpi2_ed25519 ]; then
    rm -fv ~/.ssh/known_hosts_harmgen
fi

# The kernel source tree is 1.5-2.6 GB. Uncomment to delete it too.
# rm -rf "${SRC}"

echo '==> Done.'
virsh -c qemu:///system list --all | grep -F "${DOMAIN}" || echo "    domain gone"
```

## Reproducing Section 7

Both case studies run entirely inside the VMs, from `/home/vagrant/harm-exp`. The testbed is a two-hop `veth` dumbbell in network namespaces: `htb` sets the bottleneck rate, `netem` the RTT, and `pfifo` (or `dualpi2`) the queue. `iperf3` runs in reverse mode (`-R`), so the server is the sender; `-C` propagates the CCA to both ends.

### Experiment dependencies

Run once in **each** VM:

``` bash
sudo apt-get update
sudo apt-get install -y iperf3 tcpstat net-tools tcpdump python3-pip
pip3 install numpy pandas matplotlib seaborn scikit-learn scapy ruptures psutil
```

`modAL` is vendored in `harm-exp/modAL`, so it needs no install. Verify:

``` bash
cd ~/harm-exp
python3 -c "import numpy, pandas, matplotlib, seaborn, sklearn, scapy, ruptures, psutil, modAL; print('deps ok')"
```

### Per-boot configuration

`vm-bbrv3` needs nothing beyond the dependencies — `bbr` (v3), `bbr1` and `cubic` are all built in.

`vm-dualpi2-prague` needs the L4S modules and ECN, **every boot**:

``` bash
sudo modprobe sch_dualpi2 tcp_prague
sudo sysctl -w net.ipv4.tcp_ecn=3
sudo sysctl -w net.ipv4.tcp_congestion_control=prague
```

`tcp_ecn=3` is what makes Cubic respond to ECN, which §7.2 requires for a fair comparison. This kernel extends `tcp_ecn` beyond mainline's 0–2 to 0–5, where `3` = AccECN on both incoming and outgoing connections; the default of `2` requests no ECN on *outgoing* connections, so Cubic would never negotiate it. Confirm with:

``` bash
sysctl net.ipv4.tcp_ecn net.ipv4.tcp_congestion_control
lsmod | grep -E 'dualpi2|prague'
```

To make it survive reboots:

``` bash
echo -e "sch_dualpi2\ntcp_prague" | sudo tee /etc/modules-load.d/l4s.conf
printf 'net.ipv4.tcp_ecn=3\nnet.ipv4.tcp_congestion_control=prague\n' | sudo tee /etc/sysctl.d/99-l4s.conf
```

### Section 7.1 — Cubic vs BBRv3

On **vm-bbrv3**. The paper uses Mahak here, comparing Cubic vs BBRv3 against Cubic vs BBRv1 to track what changed between releases.

``` bash
cd ~/harm-exp
./run_section7_1.sh          # Mahak, both CCA pairs, long + short flows, plus Fig 15 heatmaps
```

Or one pair at a time:

``` bash
./run_mahak.sh               # Mahak long-flow, Cubic vs BBRv1 (edit -compete_cc for bbr)
./run_harmgen_long_flow.sh   # HarmGen long-flow, Cubic vs BBRv3
./run_harmgen_short_flow.sh  # HarmGen short-flow, Cubic vs BBRv3
```

Outputs land in `./mahak_results/` (Mahak) and `./results/` (HarmGen).

### Section 7.2 — L4S: TCP Prague vs Cubic

On **vm-dualpi2-prague**, after the per-boot configuration above. The paper uses HarmGen here. `tc_single_run.py` switches the bottleneck to `dualpi2` on its own whenever either CCA is `prague`; everything else uses `pfifo`.

``` bash
cd ~/harm-exp
./run_section7_2.sh          # HarmGen, both harm directions, long + short flows
```

Or one at a time:

``` bash
./run_harmgen_long_flow.sh   # harm Prague does to Cubic (beta=cubic, alpha=prague)
./run_harmgen_short_flow.sh  # Cubic short flows vs a long Prague flow (Fig 18)
./run_mahak.sh               # Mahak long-flow, Cubic vs Prague over DualPI2
```

### Runtime and retrieving results

One long-flow evaluation is two 180 s runs (β alone, then β vs α); one short-flow evaluation is three 120 s runs. Each HarmGen script stops at whichever comes first, the experiment budget or `TIME_LIMIT`; with the shipped settings the 30-hour cap is reached first, so budget on **about 30 hours per script** and several days for all of Section 7. Use `tmux` or `nohup`.

Pull results back to the host **before** any `vagrant rsync` or `vagrant up`:

``` bash
cd vm-bbrv3            # or vm-dualpi2-prague
vagrant rsync-back
```

## Where to find the results

After `vagrant rsync-back`, everything is under the VM folder's own `harm-exp`. Each run gets a directory named `<beta_cca>-<alpha_cca>-<time_limit>-<flow_type>`, so a script's `TIME_LIMIT` is part of the path.

### Section 7.1 — Mahak results

In `vm-bbrv3/harm-exp/mahak_results/`:

| Path | What it is |
| --- | --- |
| `cubic-bbr1-240.0-long-flow/final_predictions.csv` | predicted harm for every point in the search space |
| `.../selected_samples.csv` | the settings Mahak actually chose to measure, in order |
| `.../mahak-<var>-heatmap-cubic-bbr1-long-flow.pdf` | predicted-harm heatmaps, bandwidth against RTT / queue / flow counts (paper Fig 15) |
| `.../sampling_mahak_cubic_bbr1_240.0_long-flow_<var>.pdf` | where the active learner sampled (paper Fig 8) |
| `.../experiments/<bw>bw-<rtt>rtt-<queue>q-.../` | per-experiment throughput traces and PNGs |
| `.../mahak_cubic_bbr1.log` | per-iteration log: chosen point, harm, elapsed time |
| `compare-bbr1-vs-bbr-240.0-long-flow-max-improvement/` | where BBRv3 reduces harm most (paper Fig 16) |
| `compare-bbr1-vs-bbr-240.0-short-flow-min-difference/` | where BBRv3 changes nothing (paper Fig 17) |

Each `compare-*` directory holds `selected_conditions.csv` (the conditions picked, with both CCAs' predicted harm) and one `top<N>_<condition>.pdf` per CCA, so the BBRv1 and BBRv3 figures for the same network setting sit side by side.

### Section 7.2 — HarmGen results

In `vm-dualpi2-prague/harm-exp/results/`:

| Path | What it is |
| --- | --- |
| `prague-cubic-30.0-long-flow/HarmGen_prague_cubic.csv` | every measured setting and its harm |
| `.../genetic_algorithm_prague_cubic_harm_dict_*.pckl` | the same data as a pickle, keyed by chromosome |
| `.../genetic_algorithm_prague_cubic_populations_*.pckl` | population of each generation |
| `.../top_30_harm_HarmGen_prague_cubic.png` | the 30 most harmful settings, bandwidth against RTT |
| `.../top_harm/top<N>_<condition>.pdf` | throughput over time for the highest-harm settings (paper Fig 18) |
| `.../raygen-prague-cubic-<label>/<condition>/` | per-experiment traces, PNG and logs |
| `.../ga_prague_cubic_<label>.log` | per-generation log: best / average / median harm |

`<condition>` is `<bw>bw-<rtt>rtt-<queue>q-<beta_cca>-<n_beta>-<alpha_cca>-<n_alpha>`, so the network setting of any figure can be read straight off its filename.
