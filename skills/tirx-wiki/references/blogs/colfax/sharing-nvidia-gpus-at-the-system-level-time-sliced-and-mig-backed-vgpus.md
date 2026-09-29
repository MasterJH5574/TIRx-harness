:::::::::::::::::::::::::::::::::::::::::::::::: {.wp-block-group .is-layout-flow .wp-block-group-is-layout-flow role="main" style="margin-top:var(--wp--preset--spacing--50)"}
::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained}
# Sharing NVIDIA® GPUs at the System Level: Time-Sliced and MIG-Backed vGPUs {#sharing-nvidia-gpus-at-the-system-level-time-sliced-and-mig-backed-vgpus .wp-block-post-title style="margin-bottom:var(--wp--preset--spacing--40);"}
:::

::::::::::::::::::::::::::::: {.entry-content .wp-block-post-content .has-global-padding .is-layout-constrained .wp-block-post-content-is-layout-constrained}
While some modern applications for GPUs aim to consume all GPU resources
and even scale to multiple GPUs (deep learning training, for instance),
other applications require only a fraction of GPU resources (like some
deep learning inferencing) or don't use GPUs all the time (for example,
a developer working on an NVIDIA CUDA® application may have to use the
GPU only sporadically). To better utilize the computing resources,
NVIDIA® GPUs have long been able to concurrently serve multiple
processes with time-sliced context switching and CUDA Multi-Process
Service (MPS). Although these technologies are still valid today, they
have limitations:

- the need for a shared environment (i.e., all processes sharing a GPU
  must run within the same operating system),
- no fault isolation (that is, the failure of one process can crash the
  others), and
- no quality of service (QoS) guarantees for a consistent performance.

This is where the virtual GPUs (vGPUs) come in to allow system-level
sharing of a physical GPU across multiple virtual machines (VMs), with
GPU memory partitioning and secure process isolation. Moreover, with the
introduction of the Multi-Instance GPU (MIG) technology in NVIDIA's
Ampere architecture, vGPUs received a new capability that allows for
hardware-level partitioning of GPU compute resources and quality of
service guarantees.

This publication:

1.  [Explains how MIG-capable GPUs can be used in the context of GPU
    virtualization](#sec-system-level-gpu-sharing);
2.  [Demonstrates the configuration of time-sliced and MIG-backed vGPUs
    in the Proxmox Virtual Environment](#sec-pve-with-vgpus);
3.  [Studies the performance implications of these
    approaches](#sec-performance-tests); and
4.  [Outlines the potential use cases for time-sliced and MIG-backed
    vGPU technologies](#sec-use-cases).

To skip the article and jump to the performance results, [click
here](#sec-performance-results).

## System-Level GPU Sharing {#sec-system-level-gpu-sharing .wp-block-heading}

### What is MIG? {#what-is-mig .wp-block-heading}

Multi-Instance GPU (MIG) is a feature of some NVIDIA GPUs that allows
the user to partition the GPU into multiple GPU instances (GIs), with
each instance having dedicated resources:

- memory,
- streaming multiprocessors (SMs), and
- engines (for direct memory access, encoding/decoding).

Each GPU instance can be further partitioned by the user into multiple
compute instances (CIs), which share the memory and engines of the
parent GI but have dedicated SMs.

Each CI can execute a process independently of and concurrently with all
the other CIs, which can be beneficial for increasing the GPU
utilization with multiple small and intermittent workloads. Furthermore,
the MPS technology can be used alongside MIG to run multiple processes
on each CI.

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-27.png?resize=960%2C388&amp;ssl=1"
class="wp-image-10526" data-recalc-dims="1" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-27.png?resize=960%2C388&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-27.png?resize=480%2C194&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-27.png?resize=768%2C310&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-27.png?resize=1536%2C620&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-27.png?w=1874&amp;ssl=1 1874w"
sizes="(max-width: 960px) 100vw, 960px" width="960" height="388" />
</figure>

The number and configuration of MIG instances for each GPU is determined
by the hardware organization. The `nvidia-smi` tool can report the
available GPU instance profiles with the command

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

nvidia-smi mig -lgip
```
:::

Here's a part of the output of this command obtained on a system with
NVIDIA H100 NVL GPUs:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-1.png?resize=749%2C540&amp;ssl=1"
class="wp-image-10474" data-recalc-dims="1" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-1.png?resize=749%2C540&amp;ssl=1 749w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-1.png?resize=375%2C270&amp;ssl=1 375w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-1.png?resize=768%2C553&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-1.png?w=1432&amp;ssl=1 1432w"
sizes="(max-width: 749px) 100vw, 749px" width="749" height="540" />
</figure>

The profiles have names, such as `MIG 2g.24gb`, which indicate the
number of the compute and memory slices allocated to the GPU instance of
this type. Additional information about each profile that can be
inferred from this output is:

- The number of instances of this type that can be configured,
- Memory per instance,
- Whether peer-to-peer communication is supported,
- Number of SMs and copy engines (CE) per instance,
- Number of decoder (DEC) and JPEG engines, and
- Number of encoders (ENC) and optical flow accelerator (OFA) engines.

Note that ENC and OFA features are available only in the profile
`MIG 1g.12gb+me` (as indicated by the suffix `+me`) or in the profile
`MIG 7g.94gb`) comprising all GPU slices. These media engines cannot be
shared across multiple MIG instances.

Not all MIG instances configured on a GPU have to be identical.
Combinations of different types are allowed as long as they follow the
placements dictated by the hardware design. The available placements can
be obtained with `nvidia-smi mig -lgipp`. For example, the H100 NVL GPU
supports the following:

``` wp-block-code
GPU  0 Profile ID 19 Placements: {0,1,2,3,4,5,6}:1
GPU  0 Profile ID 20 Placements: {0,1,2,3,4,5,6}:1
GPU  0 Profile ID 15 Placements: {0,2,4,6}:2
GPU  0 Profile ID 14 Placements: {0,2,4}:2
GPU  0 Profile ID  9 Placements: {0,4}:4
GPU  0 Profile ID  5 Placement : {0}:4
GPU  0 Profile ID  0 Placement : {0}:8
```

Because of memory and SM isolation, MIG instances operate independently
from each other and the workload on one instance does not impact the
performance of the others, which allows for QoS guarantees for
MIG-backed applications. There are, however, two exceptions to these
guarantees: the MIG instances share the GPU's PCIe® bus and power
resources. So, workloads dependent on host-to-device or device-to-host
transfer performance can interfere with each other when running on MIG
instances of a single GPU. And so can highly compute intensive
applications that push the electrical power available to a GPU card to
its limits. We discuss these effects [later in this
article](#sec-discussion).

### What are vGPUs? {#what-are-vgpus .wp-block-heading}

NVIDIA vGPU is a feature implemented in driver software that allows
access to a single NVIDIA GPU from multiple virtual machines. This is a
step up above the PCIe pass-through mode of GPU virtualization, in which
the entire GPU is assigned to a single VM. Indeed, vGPUs use the Single
Root I/O Virtualization (SR-IOV) functions to allow secure access to
time-sliced or partitioned GPU resources.

The vGPU technology is helpful to cloud service providers and on-premise
GPU infrastructures alike, as it allows them to offer shared access to a
GPU to independent users, each with their own environment and
applications.

To enable vGPUs on a virtual host, the administrator must install
host-side vGPU drivers on the hypervisor machine, configure VMs with
access to specific vGPUs, and install guest-side vGPU drivers in the
VMs.

NVIDIA vGPU software requires a special license. The license for the
vGPU software is available as a standalone subscription and as a part of
the NVIDIA AI Enterprise (NVAIE) software suite.

### Time-Sliced vGPUs {#time-sliced-vgpus .wp-block-heading}

Most NVIDIA GPUs support the time-sliced vGPU mode. In this mode, each
vGPU receives a dedicated share of the physical GPU memory (frame
buffer) and has access to all SMs and media engines on the GPU.

When multiple time-sliced vGPUs run processes at the same time, the vGPU
driver stack performs context switching between them according to the
policies specified by the administrator. As a result, if there is only
one active tenant, this tenant can utilize the entire GPU; if there are
multiple active tenants, they share the GPU's compute performance. Work
for a specific vGPU gets scheduled 480 or 960 times per second,
depending on the number of vGPUs configured on the physical GPU.

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-28.png?resize=960%2C540&amp;ssl=1"
class="wp-image-10527" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-28.png?resize=960%2C540&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-28.png?resize=480%2C270&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-28.png?resize=240%2C135&amp;ssl=1 240w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-28.png?resize=768%2C431&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-28.png?resize=1536%2C862&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-28.png?w=1920&amp;ssl=1 1920w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="540" />
</figure>

### MIG-Backed vGPUs {#mig-backed-vgpus .wp-block-heading}

MIG-capable NVIDIA GPUs allow MIG-backed vGPUs, which is an alternative
approach to time-sliced vGPUs. In this mode, each vGPU receives a
dedicated share of the physical GPU memory *and* a dedicated share of
the SMs (and media engines, if applicable).

When multiple MIG-backed vGPUs are simultaneously loaded, their
workloads run in parallel on the dedicated SMs and memory. As a result,
whether there is only one or multiple active tenants, each of them gets
a fixed and guaranteed fraction of the GPU's performance according to
the type of the MIG instance that the vGPU runs on (as long as the PCIe
bandwidth and the power budget do not interfere with the performance).

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-29.png?resize=960%2C540&amp;ssl=1"
class="wp-image-10528" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-29.png?resize=960%2C540&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-29.png?resize=480%2C270&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-29.png?resize=240%2C135&amp;ssl=1 240w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-29.png?resize=768%2C431&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-29.png?resize=1536%2C862&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-29.png?w=1920&amp;ssl=1 1920w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="540" />
</figure>

## Configuring a Proxmox Virtual Environment with vGPUs {#sec-pve-with-vgpus .wp-block-heading}

To demonstrate the configuration procedure for a virtual environment
with time-sliced and MIG-backed vGPUs, we use a server based on an
Intel® M50CYP2SBSTD board with an Intel® Xeon® Gold 6334Y CPU (24 cores
per socket, 2 sockets and hyper-threading for a total of 96 logical
CPUs) and two NVIDIA A30 GPUs, one GPU per socket.

### Preparing Proxmox {#preparing-proxmox .wp-block-heading}

We installed Proxmox Virtual Environment 8.2.2 with Linux kernel
6.5.13-5. Note that we have attempted to configure vGPUs with the latest
available kernel, 6.8.4-3, however, it was not compatible with NVAIE
software stack version 550.54.16, which is the latest at the time of the
writing. To configure the virtual host to boot the correct kernel
version, we edited `/etc/default/grub` and inserted

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

GRUB_DEFAULT="gnulinux-advanced-1bfdc25e-20b1-4bbf-9d9b-88feb967c8e8>gnulinux-6.5.11-8-pve-advanced-1bfdc25e-20b1-4bbf-9d9b-88feb967c8e8"
```
:::

Additionally, to enable the correct functionality of NVIDIA vGPUs, we
added the IOMMU setting to `/etc/default/grub`:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

GRUB_CMDLINE_LINUX_DEFAULT="quiet intel_iommu=on"
```
:::

and disabled the Nouveau kernel module:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

echo blacklist nouveau > /etc/modprobe.d/blacklist.conf
```
:::

After that, we updated the initramfs and the Grub boot menu and
restarted the host:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

update-initramfs -u
update-grub
reboot
```
:::

After reboot, IOMMU should be enabled and the Nouveau driver should not
be loaded:

``` wp-block-code
root@pve-1-1-dev:~# dmesg | grep "IOMMU enabled"
[    0.490456] DMAR: IOMMU enabled
root@pve-1-1-dev:~# lsmod | grep -c nouveau
0
```

Finally, we installed additional packages that the NVIDIA vGPU driver
stack expects as dependencies:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

apt install -y dkms gcc make proxmox-default-headers proxmox-headers-`uname -r`
```
:::

### Installing the Host Software {#installing-the-host-software .wp-block-heading}

We obtained NVIDIA vGPU software as a part of the NVAIE package as a
download from the [NVIDIA Licensing
Portal](https://ui.licensing.nvidia.com) (Software Downloads \> Product
Family: NVAIE \> NVIDIA AI Enterprise 5.0 Software Package for Linux
KVM).

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-22.png?resize=960%2C538&amp;ssl=1"
class="wp-image-10506" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-22.png?resize=960%2C538&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-22.png?resize=480%2C270&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-22.png?resize=240%2C135&amp;ssl=1 240w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-22.png?resize=768%2C430&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-22.png?resize=1536%2C860&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-22.png?resize=2048%2C1147&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="538" />
</figure>

After unzipping the archive, we copied the file
`Host_Drivers/NVIDIA-Linux-x86_64-550.54.16-vgpu-kvm-aie.run` to the
Proxmox host and executed it to install the host drivers:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

chmod +x NVIDIA-Linux-x86_64-550.54.16-vgpu-kvm-aie.run
./NVIDIA-Linux-x86_64-550.54.16-vgpu-kvm-aie.run --dkms
```
:::

After a successful installation, we are able to run `nvidia-smi`:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-3.png?resize=864%2C540&amp;ssl=1"
class="wp-image-10483" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-3.png?resize=864%2C540&amp;ssl=1 864w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-3.png?resize=432%2C270&amp;ssl=1 432w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-3.png?resize=768%2C480&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-3.png?resize=1536%2C960&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-3.png?w=1660&amp;ssl=1 1660w"
sizes="auto, (max-width: 864px) 100vw, 864px" width="864"
height="540" />
</figure>

This indicates a successful installation of host-side drivers.

### Configuring Time-Sliced vGPUs {#configuring-time-sliced-vgpus .wp-block-heading}

NVIDIA vGPU software comes with a script that simplifies the setup of
vGPUs by enabling the virtual functions for SR-IOV. Here is a command
that enables virtual functions on all GPUs in the virtual host using
that script:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

/usr/lib/nvidia/sriov-manage -e ALL
```
:::

This configuration is not persistent across reboots. To re-enable the
virtual functions after reboot, you can set up a `systemd` service that
runs the above command at boot.

If at a later point in time, we have to disable the virtual functions,
we can run

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

/usr/lib/nvidia/sriov-manage -d ALL
```
:::

With this done, we can create a virtual machine with a time-sliced vGPU.
From the web UI, we had to select the VM and go to Hardware \> Add \>
PCI Device:\

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-4.png?resize=960%2C415&amp;ssl=1"
class="wp-image-10484" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-4.png?resize=960%2C415&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-4.png?resize=480%2C207&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-4.png?resize=768%2C332&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-4.png?resize=1536%2C663&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-4.png?w=1968&amp;ssl=1 1968w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="415" />
</figure>

Then, under Raw Device, we can find the original device (in the
screenshot below, it is `0000:4b:00.0`) and the virtual functions
underneath it created by the `sriov-manage` tool (in the screenshot
below, the virtual functions are `0000:4b:00.4`, `0000:4b:00.5`, etc.):

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-5.png?resize=847%2C540&amp;ssl=1"
class="wp-image-10485" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-5.png?resize=847%2C540&amp;ssl=1 847w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-5.png?resize=424%2C270&amp;ssl=1 424w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-5.png?resize=768%2C490&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-5.png?resize=1536%2C979&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-5.png?w=1600&amp;ssl=1 1600w"
sizes="auto, (max-width: 847px) 100vw, 847px" width="847"
height="540" />
</figure>

After selecting the virtual function, we should go to MDev Type and
select the type of vGPU required:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-6.png?resize=960%2C470&amp;ssl=1"
class="wp-image-10486" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-6.png?resize=960%2C470&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-6.png?resize=480%2C235&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-6.png?resize=768%2C376&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-6.png?resize=1536%2C752&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-6.png?w=1908&amp;ssl=1 1908w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="470" />
</figure>

The nomenclature for time-sliced vGPUs has the physical GPU model (in
our example, A30) followed by the amount of framebuffer (i.e., GPU
memory) allocated to it.

To configure additional vGPUs in the same or VM or others, we must
select other virtual functions. The types of vGPUs available for the
chosen virtual function is automatically reflected in the interface.

### Configuring MIG-Backed vGPUs {#configuring-mig-backed-vgpus .wp-block-heading}

MIG-backed vGPUs are configured similarly to time-sliced, with
additional steps that enable MIG on the physical host, sets up GPU
instance profiles, and creates compute instance profiles.

First, to enable the virtual functions for SR-IOV, we ran:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

/usr/lib/nvidia/sriov-manage -e ALL
```
:::

Second, to enable MIG on all physical GPUs, use `nvidia-smi` on the
virtual host, we ran:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

nvidia-smi -mig 1
```
:::

This reports the list of GPUs and their addresses followed by "all
done":

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-7.png?resize=960%2C133&amp;ssl=1"
class="wp-image-10488" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-7.png?resize=960%2C133&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-7.png?resize=480%2C66&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-7.png?resize=768%2C106&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-7.png?w=1130&amp;ssl=1 1130w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="133" />
</figure>

Finally, we had to choose and create GPU instance profiles. The list of
GPU instance profiles can be queried with `nvidia-smi -lgip`:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-8.png?resize=960%2C530&amp;ssl=1"
class="wp-image-10489" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-8.png?resize=960%2C530&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-8.png?resize=480%2C265&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-8.png?resize=768%2C424&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-8.png?w=1430&amp;ssl=1 1430w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="530" />
</figure>

The nomenclature for these profiles is the number of GPU slices (e.g.,
above, `1g` corresponds to one slice comprising 14 SMs, `2g` to two
slices comprising 28 SMs, etc.) followed by the amount of the
framebuffer (e.g., `6gb` corresponds to 6 GB of GPU memory). The `+me`
suffix indicates the allocation of media engines to the GPU instance
profile.

To create GPU instance profiles, we ran the following command with a
list the profiles in the order corresponding to their placement, e.g.:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-13.png?resize=960%2C128&amp;ssl=1"
class="wp-image-10494" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-13.png?resize=960%2C128&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-13.png?resize=480%2C64&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-13.png?resize=768%2C102&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-13.png?w=1504&amp;ssl=1 1504w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="128" />
</figure>

Finally, we created compute instances on the GPU instances. This allows
us to use the full GPU instance for the vGPU or further subdivide the
GPU instance into smaller compute instances with a shared framebuffer.
The list of compute instance profiles can be obtained with `nvidia-smi`:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-14.png?resize=960%2C455&amp;ssl=1"
class="wp-image-10495" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-14.png?resize=960%2C455&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-14.png?resize=480%2C227&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-14.png?resize=768%2C364&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-14.png?resize=1536%2C728&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-14.png?w=1596&amp;ssl=1 1596w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="455" />
</figure>

This nomenclature has the prefix ending in `-c`, which indicates the
number of compute slices in the compute instance. Here's an example of
creating the `2g.12gb` compute instance in all MIG slices on all GPUs:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-15.png?resize=960%2C102&amp;ssl=1"
class="wp-image-10496" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-15.png?resize=960%2C102&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-15.png?resize=480%2C51&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-15.png?resize=768%2C82&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-15.png?resize=1536%2C163&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-15.png?w=1922&amp;ssl=1 1922w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="102" />
</figure>

The list of currently running MIG configuration in the virtual host is
now included in the output of `nvidia-smi`:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-16.png?resize=600%2C540&amp;ssl=1"
class="wp-image-10497" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-16.png?resize=600%2C540&amp;ssl=1 600w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-16.png?resize=300%2C270&amp;ssl=1 300w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-16.png?resize=768%2C691&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-16.png?resize=1536%2C1382&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-16.png?w=1652&amp;ssl=1 1652w"
sizes="auto, (max-width: 600px) 100vw, 600px" width="600"
height="540" />
</figure>

To make this configuration persistent across host reboots, similarly to
time-sliced vGPUs, one must configure a `systemd` service that executes
a script with all of the necessary commands, e.g.,

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

/usr/lib/nvidia/sriov-manage -e ALL
/usr/bin/nvidia-smi -mig 1
/usr/bin/nvidia-smi mig -cgi 2g.12gb,2g.12gb
/usr/bin/nvidia-smi mig -cci 2g.12gb
```
:::

If at some point we need to disable the compute instances, GPU
instances, or disable MIG completely, we can use:

``` wp-block-code
nvidia-smi mig -dci # Delete compute instances
nvidia-smi mig -dgi # Delete GPU instances
nvidia-smi -mig 0   # Disable MIG
```

With the host configured, we can proceed to add a MIG-backed vGPU to a
virtual machine. The first step is the same as for time-sliced vGPUs.
That is, in the Proxmox web UI, select the VM, go to Hardware, and
choose Add a PCI Device:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-4.png?resize=960%2C415&amp;ssl=1"
class="wp-image-10484" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-4.png?resize=960%2C415&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-4.png?resize=480%2C207&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-4.png?resize=768%2C332&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-4.png?resize=1536%2C663&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-4.png?w=1968&amp;ssl=1 1968w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="415" />
</figure>

The next step is also similar to time-sliced vGPU setup: choose Raw
Device and select a virtual function on the physical GPU:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-5.png?resize=847%2C540&amp;ssl=1"
class="wp-image-10485" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-5.png?resize=847%2C540&amp;ssl=1 847w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-5.png?resize=424%2C270&amp;ssl=1 424w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-5.png?resize=768%2C490&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-5.png?resize=1536%2C979&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-5.png?w=1600&amp;ssl=1 1600w"
sizes="auto, (max-width: 847px) 100vw, 847px" width="847"
height="540" />
</figure>

And for the last step, select one of the available instance types:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-17.png?resize=960%2C434&amp;ssl=1"
class="wp-image-10498" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-17.png?resize=960%2C434&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-17.png?resize=480%2C217&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-17.png?resize=768%2C348&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-17.png?resize=1536%2C695&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-17.png?w=1958&amp;ssl=1 1958w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="434" />
</figure>

Note that the nomenclature for vGPU types for MIG-backed GPUs has an
additional number compared to the time-sliced GPUs (e.g., `A30-2-12C`
instead of `A30-12C`), which indicates the number of GPU compute slices
dedicated to the vGPU.

### Installing the Guest Software {#installing-the-guest-software .wp-block-heading}

With a time-sliced or MIG-backed vGPU configured in the VM, we can boot
the guest. In our case, the guest OS is Ubuntu 22.04 with no
modifications.

Before installing the vGPU drivers, we disabled the nouveau driver in
the guest OS and rebooted the guest:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

echo blacklist nouveau > /etc/modprobe.d/blacklist-nouveau.conf
echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nouveau.conf
update-initramfs -u
reboot
```
:::

After that, we revisited the NVAIE archive downloaded from the NVIDIA
Lincensing Portal and copied the file
`Guest_Drivers/nvidia-linux-grid-550_550.54.15_amd64.deb` to the guest
OS and installed it:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

apt-get install nvidia-linux-grid-550_550.54.15_amd64.deb
```
:::

Now, `nvidia-smi` inside the guest displays the vGPU configured
according to the process outlined above:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-18.png?resize=701%2C540&amp;ssl=1"
class="wp-image-10500" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-18.png?resize=701%2C540&amp;ssl=1 701w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-18.png?resize=351%2C270&amp;ssl=1 351w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-18.png?resize=768%2C591&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-18.png?resize=1536%2C1183&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-18.png?w=1670&amp;ssl=1 1670w"
sizes="auto, (max-width: 701px) 100vw, 701px" width="701"
height="540" />
</figure>

Additional `nvidia-smi` commands demonstrate the GPU instance profile
and compute instance profile:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-19.png?resize=960%2C458&amp;ssl=1"
class="wp-image-10501" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-19.png?resize=960%2C458&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-19.png?resize=480%2C229&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-19.png?resize=768%2C367&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-19.png?w=1508&amp;ssl=1 1508w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="458" />
</figure>

Note that, while the GPU instance profile must be configured on the
virtual host, the compute instance profile may be configured on the host
as well as on the guest.

### Configuring vGPU License {#configuring-vgpu-license .wp-block-heading}

Even if there is no active license for the vGPU software running on the
guest, the virtual GPU driver on the guest should load and display the
configuration, and applications using the GPU may run for a limited
period of time. However, we may find a message like this in
`/var/log/syslog`:

``` wp-block-code
May 22 02:46:38 cexp nvidia-gridd: Valid GRID license not found. GPU features and performance are restricted. To enable full functionality please configure licensing details.
```

Additionally, after some time, the GPU performance will be significantly
throttled.

To keep the vGPU running, we need to set up a license server, obtain a
client token, and hand this token to the `nvidia-gridd` service.

Two options are available for the license server:

1.  Cloud License Server (CLS) --- with this option, the server is
    hosted in the cloud by NVIDIA. The clients (VMs) communicate with it
    over the Internet.
2.  Delegated License Server (DLS) --- a VM or container hosted by the
    user on-premise. The clients may communicate with it over the
    private network.

For our experiments, we configured a CLS using the NVIDIA Licensing
Portal. We started at the Create Server page and followed its prompts to
create a server instance and assign entitlements to it.

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-20.png?resize=958%2C540&amp;ssl=1"
class="wp-image-10502" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-20.png?resize=958%2C540&amp;ssl=1 958w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-20.png?resize=480%2C270&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-20.png?resize=240%2C135&amp;ssl=1 240w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-20.png?resize=768%2C433&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-20.png?resize=1536%2C866&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-20.png?resize=2048%2C1154&amp;ssl=1 2048w"
sizes="auto, (max-width: 958px) 100vw, 958px" width="958"
height="540" />
</figure>

After configuring server, we used the Actions menu and selected
"Generate client config token". The browser then downloads the token as
a text file.

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/Screenshot-2024-05-28-at-4.08.52%E2%80%AFPM.png?resize=960%2C457&amp;ssl=1"
class="wp-image-10505" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/Screenshot-2024-05-28-at-4.08.52%E2%80%AFPM.png?resize=960%2C457&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/Screenshot-2024-05-28-at-4.08.52%E2%80%AFPM.png?resize=480%2C228&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/Screenshot-2024-05-28-at-4.08.52%E2%80%AFPM.png?resize=768%2C365&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/Screenshot-2024-05-28-at-4.08.52%E2%80%AFPM.png?resize=1536%2C731&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/Screenshot-2024-05-28-at-4.08.52%E2%80%AFPM.png?resize=2048%2C974&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="457" />
</figure>

To license the vGPU software on our VM, we had to copy the token file to
`/etc/nvidia/ClientConfigToken/` and restart the `nvidia-gridd` service:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

cp client_configuration_token_05-22-2024-12-35-06.tok /etc/nvidia/ClientConfigToken/
systemctl restart nvidia-gridd.service
```
:::

After that, the status of the `nvidia-gridd` service shows a successful
activation status:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-21.png?resize=960%2C313&amp;ssl=1"
class="wp-image-10503" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-21.png?resize=960%2C313&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-21.png?resize=480%2C156&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-21.png?resize=768%2C250&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-21.png?resize=1536%2C501&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-21.png?resize=2048%2C667&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="313" />
</figure>

The licensing portal reflects the active license leases:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/Screenshot-2024-05-28-at-4.16.00%E2%80%AFPM.png?resize=917%2C540&amp;ssl=1"
class="wp-image-10504" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/Screenshot-2024-05-28-at-4.16.00%E2%80%AFPM.png?resize=917%2C540&amp;ssl=1 917w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/Screenshot-2024-05-28-at-4.16.00%E2%80%AFPM.png?resize=459%2C270&amp;ssl=1 459w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/Screenshot-2024-05-28-at-4.16.00%E2%80%AFPM.png?resize=768%2C452&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/Screenshot-2024-05-28-at-4.16.00%E2%80%AFPM.png?resize=1536%2C904&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/Screenshot-2024-05-28-at-4.16.00%E2%80%AFPM.png?resize=2048%2C1206&amp;ssl=1 2048w"
sizes="auto, (max-width: 917px) 100vw, 917px" width="917"
height="540" />
</figure>

## Performance Tests {#sec-performance-tests .wp-block-heading}

To illustrate the behavior of time-sliced and MIG-backed vGPUs, we used
the NVIDIA A30 GPU and compared the following vGPU configurations:

1.  Four (4) time-sliced NVIDIA A30-6G vGPUs.
2.  Four (4) MIG-backed NVIDIA A30-1G-6GB vGPUs.

For these cases, we ran four workloads:

1.  PCIe bandwidth test;
2.  GPU memory bandwidth test;
3.  Small GEMM (general matrix-matrix multiplication) on tensor cores;
4.  Large GEMM on tensor cores.

The PCIe and GPU memory bandwidth tests illustrate the sharing of the
PCIe and GPU bandwidth. The small GEMM illustrates a problem that is too
small for the entire GPU. The large GEMM illustrates a problem that is
large enough to saturate the entire GPU.

For 4-vGPU configurations, we ran the workloads in two modes:

1.  Idle system mode: we ran the workload on only one vGPU out of four
    vGPUs.
2.  Loaded system mode: we ran the same workload on each of the four
    vGPUs.

For the loaded system mode, we launched the test on four VMs
concurrently.

### vGPU Configuration {#vgpu-configuration .wp-block-heading}

For all tests, we used the NVIDIA A30 GPU local to NUMA node 0 on our
platform. The affinity of NVIDIA GPUs can be queried on the virtual host
by running `lspci -v` (we can also use the flag "`-d 10de:`" to list
just NVIDIA devices):

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-23.png?resize=960%2C145&amp;ssl=1"
class="wp-image-10508" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-23.png?resize=960%2C145&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-23.png?resize=480%2C73&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-23.png?resize=768%2C116&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-23.png?w=1388&amp;ssl=1 1388w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="145" />
</figure>

We pinned the VMs to the CPU cores belonging to the same NUMA node. The
mapping of CPU cores to NUMA nodes is available in the output of
`lspci`:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-24.png?resize=960%2C127&amp;ssl=1"
class="wp-image-10509" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-24.png?resize=960%2C127&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-24.png?resize=480%2C64&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-24.png?resize=768%2C102&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-24.png?w=1282&amp;ssl=1 1282w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="127" />
</figure>

So we configured our four VMs to be pinned to NUMA node 0 consecutive
cores: VM1 to {0-5,48-53}, VM2 to {6-11,54-59}, VM3 to {12-17,60-65},
and VM4 to {18-23, 66-71}.

For time-sliced vGPUs, we chose devices of type A30-6C:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-26.png?resize=960%2C528&amp;ssl=1"
class="wp-image-10512" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-26.png?resize=960%2C528&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-26.png?resize=480%2C264&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-26.png?resize=768%2C423&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-26.png?resize=1536%2C846&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-26.png?w=1962&amp;ssl=1 1962w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="528" />
</figure>

For MIG-backed vGPUs, we used devices of type A30-1-6C and configured
four GPU instances of type 1g.6gb. Inside each GPU instance, we created
a single compute instance of the same kind.

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

nvidia-smi mig -cgi 1g.6gb,1g.6gb,1g.6gb,1g.6gb
nvidia-smi mig -cci 1g.6gb
```
:::

<figure class="wp-block-image size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-39.png?resize=960%2C528&amp;ssl=1"
class="wp-image-10551" style="width:650px;height:auto"
data-recalc-dims="1" loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-39.png?resize=960%2C528&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-39.png?resize=480%2C264&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-39.png?resize=768%2C422&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-39.png?w=1050&amp;ssl=1 1050w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="528" />
</figure>

### Test Workloads {#test-workloads .wp-block-heading}

#### PCIe Bandwidth {#pcie-bandwidth .wp-block-heading}

We measured the PCIe bandwidth using the `bandwidthTest` sample from
NVIDIA's [CUDA Samples](https://github.com/NVIDIA/cuda-samples). To make
the test run long enough, we set value of `MEMCOPY_ITERATIONS` to 10000
inside `bandwidthTest.cu`:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

#define MEMCOPY_ITERATIONS 10000
```
:::

After that, we recompiled the code. The test invocation command for
device-to-host bandwidth is:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

./bandwidthTest --dtoh --mode=range --start=32000000 --end=32000000 --increment=1
```
:::

and for host-to-device bandwidth:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

./bandwidthTest --htod --mode=range --start=32000000 --end=32000000 --increment=1
```
:::

We report the host-to-device (HtoD) and device-to-host (DtoH)
performance values separately.

#### GPU Memory Bandwidth {#gpu-memory-bandwidth .wp-block-heading}

For GPU memory bandwidth measurement, we used the same `bandwidthTest`
sample but different arguments. We also set `MEMCOPY_ITERATIONS` in
`bandwidthTest.cu` to 10000:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

#define MEMCOPY_ITERATIONS 10000
```
:::

Then we recompile the code and execute the test with

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

./bandwidthTest --dtod --mode=range --start=384000000 --end=384000000 --increment=1
```
:::

The performance result in is the bandwidth reported by the application.

#### Small GEMM {#small-gemm .wp-block-heading}

The small GEMM workload is representative of calculations that are
compute-bound but that do not have enough parallelism to scale across
the entire GPU. Some cases of small-batch deep learning inferencing fall
into this category.

To run a small GEMM, we used the profiler tool from NVIDIA's [CUTLASS
collection](https://github.com/NVIDIA/cutlass). We chose the TF32 data
type to invoke the tensor cores in the NVIDIA A30 GPU and to emulate an
AI workload. The invocation command is:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

./cutlass_profiler --operation=Gemm --m=1024 --n=1024 --k=1024 --A=tf32:row --B=tf32:row --C=tf32 --verification-enabled=false --profiling-iterations=100000
```
:::

This corresponds to the
`cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_tt_align4` operation
in CUTLASS.

#### Large GEMM {#large-gemm .wp-block-heading}

The large GEMM workload is representative of compute-bound calculations
that are large enough to scale across the entire GPU (think deep
learning training).

Similarly to the small GEMM case, we used the CUTLASS profiler tool to
run this workload, invoking it with the following command:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .title: .; .notranslate title=""}

./cutlass_profiler --operation=Gemm --m=8192 --n=8192 --k=16400 --A=tf32:row --B=tf32:row --C=tf32 --verification-enabled=false --profiling-iterations=100
```
:::

This corresponds to the
`cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_tt_align4` operation
in CUTLASS.

### Results {#sec-performance-results .wp-block-heading}

The table below summarizes the measured performance of the two
configurations (time-sliced versus MIG-backed vGPUs) for the four
workloads (PCIe and memory bandwidth, small GEMM, and large GEMM) in the
two modes (idle and loaded). The reported performance numbers are
rounded to 3 significant figures (where available). For the loaded case,
the table reports the performance of one vGPU, averaged over the four
VMs (rather than the sum of all vGPU performance numbers).

<figure class="wp-block-table">
<table>
<thead>
<tr>
<th>Workload</th>
<th>Mode</th>
<th>Time-sliced vGPUs</th>
<th>MIG-Backed vGPUs</th>
</tr>
</thead>
<tbody>
<tr>
<td>PCIe bandwidth HtoD</td>
<td>Idle</td>
<td>25.2 GB/s</td>
<td>25.2 GB/s</td>
</tr>
<tr>
<td>PCIe bandwidth HtoD</td>
<td>Loaded</td>
<td>6.3 GB/s</td>
<td>6.3 GB/s</td>
</tr>
<tr>
<td>PCIe bandwidth DtoH</td>
<td>Idle</td>
<td>26.2 GB/s</td>
<td>26.3 GB/s</td>
</tr>
<tr>
<td>PCIe bandwidth DtoH</td>
<td>Loaded</td>
<td>6.6 GB/s</td>
<td>6.6 GB/s</td>
</tr>
<tr>
<td>Memory bandwidth</td>
<td>Idle</td>
<td>786 GB/s</td>
<td>196 GB/s</td>
</tr>
<tr>
<td>Memory bandwidth</td>
<td>Loaded</td>
<td>160 GB/s</td>
<td>196 GB/s</td>
</tr>
<tr>
<td>Small GEMM in TF32</td>
<td>Idle</td>
<td>6680 GFLOP/s</td>
<td>6550 GFLOP/s</td>
</tr>
<tr>
<td>Small GEMM in TF32</td>
<td>Loaded</td>
<td>1410 GFLOP/s</td>
<td>6550 GFLOP/s</td>
</tr>
<tr>
<td>Large GEMM in TF32</td>
<td>Idle</td>
<td>68800 GFLOP/s</td>
<td>16000 GFLOP/s</td>
</tr>
<tr>
<td>Large GEMM in TF32</td>
<td>Loaded</td>
<td>14500 GFLOP/s</td>
<td>15300 GFLOP/s</td>
</tr>
</tbody>
</table>
</figure>

For context, the theoretical peak performance metrics of NVIDIA A30 GPUs
are:

- PCIe bandwidth (each direction): 32 GB/s;
- Memory bandwidth: 933 GB/s;
- TF32 Tensor Core performance: 82 TFLOP/s.

The plots below show these performance numbers with the addition of:

- The cumulative performance in the loaded mode and
- The efficiency calculated as the ratio of the observed performance to
  the theoretical peak value.

PCIe bandwidth:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-36.png?resize=960%2C399&amp;ssl=1"
class="wp-image-10539" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-36.png?resize=960%2C399&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-36.png?resize=480%2C199&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-36.png?resize=768%2C319&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-36.png?w=1119&amp;ssl=1 1119w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="399" />
</figure>

GPU memory bandwidth:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-33.png?resize=960%2C396&amp;ssl=1"
class="wp-image-10536" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-33.png?resize=960%2C396&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-33.png?resize=480%2C198&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-33.png?resize=768%2C317&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-33.png?w=1122&amp;ssl=1 1122w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="396" />
</figure>

Small GEMM performance:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-38.png?resize=960%2C397&amp;ssl=1"
class="wp-image-10548" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-38.png?resize=960%2C397&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-38.png?resize=480%2C198&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-38.png?resize=768%2C317&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-38.png?resize=1536%2C635&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-38.png?resize=2048%2C846&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="397" />
</figure>

Large GEMM performance:

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-37.png?resize=960%2C400&amp;ssl=1"
class="wp-image-10540" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-37.png?resize=960%2C400&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-37.png?resize=480%2C200&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-37.png?resize=768%2C320&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/05/image-37.png?w=1121&amp;ssl=1 1121w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="400" />
</figure>

### Discussion {#sec-discussion .wp-block-heading}

From the data reported in the previous section, we observe the following
trends in our NVIDIA A30 GPU shared across four VMs with the help of
time-sliced and MIG-backed vGPUs:

1.  When multiple vGPUs are accessing the PCIe bus, its bandwidth is
    shared between the vGPUs (6.3 GB/s per vGPU in the loaded mode
    versus 25.2 GB/s in the idle mode for host-to-device bandwidth and a
    similar split for device-to-host).
2.  If only one vGPU is using the PCIe bus, it has access to its full
    bandwidth.
3.  GPU memory bandwidth and GPU compute performance behave similarly
    when the system is idle:
    - One time-sliced vGPU stressing the GPU in a otherwise idle system
      gets access to the full bandwidth or full compute performance
      (68800 GFLOP/s in our test) of the GPU.
    - One MIG-backed vGPU stressing the GPU in an otherwise idle system
      gets only its proportional share of the full performance (196 GB/s
      is a quarter of 786 GB/s and 16000 GFLOP/s is 7% shy of a quarter
      of 68800 GFLOP/s).
4.  GPU memory bandwidth and GPU compute performance behave similarly
    when the system is loaded:
    - Each of the time-sliced vGPUs gets a fraction of the full GPU
      performance because each vGPU uses the *entire* memory and the
      *entire* compute performance *part of the time*.
    - Each of the MIG-backed vGPUs gets a fraction of the full GPU
      performance because each vGPU uses a *part of* the memory
      bandwidth and a *fraction of* the SMs *all of the time*.
    - The time-shared configuration in a loaded system has a performance
      penalty for context switching compared to the MIG configuration.
      This penalty is \~20% for our GPU bandwidth test (160 GB/s is
      \~20% lower than 196 GB/s) and \~5% of the large GEMM test (14500
      GFLOP/s is \~5% lower than 15300 GFLOP/s).
5.  The behavior of small GEMM performance deserves a special mention
    because in a loaded system, the time-sliced configuration has a
    significantly lower performance (4.6x slower in our test) than
    MIG-backed. The explanation of this observation is straightforward:
    - On a MIG-backed system, each vGPU's small workload utilizes a
      small fraction of the GPU's SMs *all of the time* but
    - On a time-sliced system, each vGPU's small workload utilizes the
      same small fraction of the GPU's SMs but only *part of the time*.
6.  It is also worth noting that the large GEMM stressed our GPU enough
    that a MIG-backed vGPU had \~5% lower performance in the loaded mode
    than in an idle system (15300 GFLOP/s versus 16000 GFLOP/s). We
    attribute this difference to the limits on the electrical power that
    the GPU allows in before it throttles. In fact, when we ran the same
    test using only 3 out of the 4 MIG-backed vGPUs, each of them
    delivered a non-throttled performance of 16000 GFLOP/s.

## Use Cases {#sec-use-cases .wp-block-heading}

The performance measurements reported above allow us to draw general
conclusions regarding the best use cases for MIG-backed and time-sliced
vGPUs.

Time-sliced vGPUs are helpful for dividing a physical GPU between
multiple tenants when:

1.  We want each tenant to be able to opportunistically burst to the
    full performance of the GPU when all other tenants are idle or
2.  We want each tenant to see the full list of SMs for the purpose of
    devising code parallelization strategies scalable to the full GPU.

Importantly, in order to use time-sliced vGPUs, these tenants must not
have any expectation of a predictable quality of service or consistent
performance because their experience will depend on the other tenants'
activity.

Examples of suitable applications of the time-sliced vGPU configuration
are:

- Multiple GPU application developers may benefit from sharing a GPU
  through the time-sliced vGPU configuration, assuming that they don't
  have to run their code for extended periods of time.
- Cloud-based deep learning inferencing applications that prioritize low
  latency of a single request over the throughput of multiple requests
  *and* expect to have a significant GPU under-utilization on average.

In most other cases, MIG-backed vGPUs will have an advantage. It is
reasonable to use MIG-backed vGPUs when:

1.  Tenants need performance guarantees and QoS for compute-bound and
    bandwidth-bound applications;
2.  The priority is to maximize the cumulative performance of all vGPUs
    by eliminating the context switching overhead of time-sliced
    configurations;
3.  The workloads of all tenants are too small to utilize the full GPU
    and the objective is to increase the utilization of the GPU's
    compute capabilities.

Examples of applications that may benefit from MIG-backed vGPUs are:

- Multiple GPU application developers working on performance
  optimization in GPU code and requiring consistent run-to-run
  performance.
- Cloud-based deep learning inferencing systems that must maximize the
  cumulative throughput of multiple concurrent inference streams.

In conclusion, we would like to reiterate that the vGPU method of
sharing a GPU across multiple tenants is a technique for system-level
sharing of hardware across virtual machines. It may be used in tandem
with hardware-level sharing techniques (MIG compute instances inside a
VM) and process-level (time slicing or MPS) sharing to obtain the
correct isolation, granularity, and interconnectedness of a GPU-based
computing system.

::::: {.sharedaddy .sd-sharing-enabled}
:::: {.robots-nocontent .sd-block .sd-social .sd-social-icon-text .sd-sharing}
### Share this: {#share-this .sd-title}

::: sd-content
- [[Share on LinkedIn (Opens in new window)]{#sharing-linkedin-10454
  hidden=""}
  LinkedIn](https://research.colfax-intl.com/sharing-nvidia-gpus-at-the-system-level-time-sliced-and-mig-backed-vgpus/?share=linkedin){.share-linkedin
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-linkedin-10454" target="_blank"
  aria-labelledby="sharing-linkedin-10454"}
- [[Share on X (Opens in new window)]{#sharing-twitter-10454 hidden=""}
  X](https://research.colfax-intl.com/sharing-nvidia-gpus-at-the-system-level-time-sliced-and-mig-backed-vgpus/?share=twitter){.share-twitter
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-twitter-10454" target="_blank"
  aria-labelledby="sharing-twitter-10454"}
- [[Share on Facebook (Opens in new window)]{#sharing-facebook-10454
  hidden=""}
  Facebook](https://research.colfax-intl.com/sharing-nvidia-gpus-at-the-system-level-time-sliced-and-mig-backed-vgpus/?share=facebook){.share-facebook
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-facebook-10454" target="_blank"
  aria-labelledby="sharing-facebook-10454"}
- [[Share on Reddit (Opens in new window)]{#sharing-reddit-10454
  hidden=""}
  Reddit](https://research.colfax-intl.com/sharing-nvidia-gpus-at-the-system-level-time-sliced-and-mig-backed-vgpus/?share=reddit){.share-reddit
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-reddit-10454" target="_blank"
  aria-labelledby="sharing-reddit-10454"}
-
:::
::::
:::::
:::::::::::::::::::::::::::::

::::::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained style="margin-top:48px;margin-bottom:48px;padding-top:5px;padding-bottom:5px"}

------------------------------------------------------------------------

### Discover more from Colfax Research {#discover-more-from-colfax-research .wp-block-heading .has-text-align-center style="margin-top:4px;margin-bottom:10px"}

Subscribe to get the latest posts sent to your email.

:::::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-container-core-group-is-layout-b821fca1 .wp-block-group-is-layout-constrained}
::::: {.wp-block-jetpack-subscriptions__supports-newline .wp-block-jetpack-subscriptions}
:::: {.wp-block-jetpack-subscriptions__container .is-not-subscriber}
::: wp-block-jetpack-subscriptions__form-elements
Type your email...

Subscribe
:::
::::
:::::
::::::
:::::::

:::::::::: wp-block-template-part
::: {.wp-block-spacer style="height:0" aria-hidden="true"}
:::

:::::::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained style="margin-top:var(--wp--preset--spacing--70)"}
::::::: {.wp-block-columns .alignwide .has-small-font-size .is-layout-flex .wp-container-core-columns-is-layout-7495e5c1 .wp-block-columns-is-layout-flex style="margin-top:var(--wp--preset--spacing--30)"}
:::::: {.wp-block-column .is-layout-flow .wp-container-core-column-is-layout-47e5a185 .wp-block-column-is-layout-flow}
::::: {.wp-block-group .is-layout-flex .wp-container-core-group-is-layout-f0ee7b9b .wp-block-group-is-layout-flex}
Posted

::: wp-block-post-date
May 29, 2024
:::

in

::: {.taxonomy-category .wp-block-post-terms}
[Article](https://research.colfax-intl.com/category/article/){rel="tag"}[,
]{.wp-block-post-terms__separator}[Benchmarks](https://research.colfax-intl.com/category/papers/benchmarks/){rel="tag"}[,
]{.wp-block-post-terms__separator}[Blog](https://research.colfax-intl.com/category/blog/){rel="tag"}[,
]{.wp-block-post-terms__separator}[HPC System
Administration](https://research.colfax-intl.com/category/papers/system-administration/){rel="tag"}[,
]{.wp-block-post-terms__separator}[Publications](https://research.colfax-intl.com/category/papers/){rel="tag"}[,
]{.wp-block-post-terms__separator}[Technology
Exploration](https://research.colfax-intl.com/category/papers/technology/){rel="tag"}
:::
:::::
::::::
:::::::
::::::::
::::::::::

:::::: {.section .wp-block-template-part}
::::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-container-core-group-is-layout-a666d811 .wp-block-group-is-layout-constrained style="padding-top:var(--wp--preset--spacing--40);padding-right:var(--wp--preset--spacing--40);padding-bottom:var(--wp--preset--spacing--40);padding-left:var(--wp--preset--spacing--40)"}
:::: wp-block-comments
## Comments {#comments .wp-block-heading}

::: {#respond .comment-respond .wp-block-post-comments-form}
### Leave a Reply [[Cancel reply](/sharing-nvidia-gpus-at-the-system-level-time-sliced-and-mig-backed-vgpus/#respond){#cancel-comment-reply-link rel="nofollow" style="display:none;"}]{.small} {#reply-title .comment-reply-title}

[Your email address will not be published.]{#email-notes} [Required
fields are marked [\*]{.required}]{.required-field-message}

Comment [\*]{.required}

Name

Email

Website

Save my name, email, and website in this browser for the next time I
comment.

Δ
:::
::::
:::::
::::::
::::::::::::::::::::::::::::::::::::::::::::::::
