"""
Real NAKIVO Backup & Replication troubleshooting documents, sourced from
NAKIVO's official Help Center (helpcenter.nakivo.com), paraphrased in
original wording. To be added to data/raw/nakivo/ to fill out the
under-represented "backup" category for the QLoRA classifier training set.
"""

import json
import os

NAKIVO_DOCS = [
    {
        "id": "nakivo-transporter-connection-lost",
        "title": "NAKIVO: Transporter Connection Lost During Backup Job",
        "url": "https://helpcenter.nakivo.com/Knowledge-Base/Content/Director-and-Transporter-Issues/Transporter-Connection-Lost.htm",
        "category": "Backup & Disaster Recovery",
        "content": """# NAKIVO: Transporter Connection Lost During Backup Job

## Problem
A backup job fails with an error indicating that the connection to a source
or target Transporter has been lost, and the job cannot maintain a stable
link between the two Transporters throughout the run.

## Root Causes
- Network instability or intermittent connectivity between the source and
  target Transporter hosts.
- The Transporter service itself has stopped or become unresponsive on one
  of the hosts.
- Firewall rules or NAT configuration blocking the required Transporter
  communication ports mid-job.

## Remediation Steps

### Step 1: Verify Transporter Services Are Running
Check that the Transporter service is active on both the source and target
machines. Restart the service on whichever host shows it stopped or
unresponsive.

### Step 2: Confirm Network Reachability
Test connectivity between the source and target Transporter hosts (ping,
and confirm the required TCP ports are open and not blocked by a firewall
along the path).

### Step 3: Re-run the Job
Once services are confirmed running and connectivity is verified, retry the
backup job.

### Step 4: Escalate if Unresolved
If the connection continues to drop after verifying services and network
connectivity, generate a support bundle from within NAKIVO Backup &
Replication and escalate to NAKIVO support for deeper investigation.
""",
    },
    {
        "id": "nakivo-transporter-connection-issues-ip-port",
        "title": "NAKIVO: Transporter Connection Errors (IP/Port Misconfiguration)",
        "url": "https://helpcenter.nakivo.com/Knowledge-Base/Content/Frequently-Asked-Questions/Miscellaneous/Transporter-Connection-Issues.htm",
        "category": "Backup & Disaster Recovery",
        "content": """# NAKIVO: Transporter Connection Errors (IP/Port Misconfiguration)

## Problem
NAKIVO Backup & Replication fails to connect to a Transporter, commonly
reporting that no response was received at the configured host/IP and port.

## Common Causes and Fixes

### No response at the configured host:port
Double-check that the IP address (or hostname) and port entered for the
Transporter are correct, and confirm the host is powered on with the
Transporter service running and actively listening on that port.

### Firewall blocking the connection
Ensure the path between the NAKIVO Director service and the Transporter's
IP/port is not blocked by a firewall on either end.

### Outdated Transporter software
An outdated Transporter version can also cause connection failures — update
the Transporter to match the current product version.

### Transporter not yet registered
Confirm the Transporter has actually been added to the product in the
Director's configuration before assuming a network-layer problem.

## When to Escalate
If none of the above resolves the issue, create a support bundle and
contact NAKIVO support directly.
""",
    },
    {
        "id": "nakivo-proxy-transporter-connection-problem",
        "title": "NAKIVO: Proxy Transporter Connection Problem (ActionException) in Multi-Network Environments",
        "url": "https://helpcenter.nakivo.com/Knowledge-Base/Content/Director-and-Transporter-Issues/Proxy-Transporter-Connection-Problem-(ActionException).htm",
        "category": "Backup & Disaster Recovery",
        "content": """# NAKIVO: Proxy Transporter Connection Problem (ActionException)

## Problem
In an environment with source and target proxy Transporters on separate
network segments, a backup or replication job fails with a
"Proxy transporter connection problem (ActionException)" error, reporting
that the connection to a transporter has been lost.

## Root Cause
This typically occurs when the source and target Transporters cannot reach
each other over their internal (inner) IP addresses — for example, when
the NAKIVO Director sits in an isolated network segment and can only reach
both Transporters via their external IPs, while the Transporters themselves
attempt to communicate directly over internal addresses that aren't mutually
routable.

## Remediation
Reconfigure the network topology so that the Director, the source
Transporter (with the source hypervisor/vCenter attached), and the target
Transporter (with the backup repository attached) can all reach each other
consistently — ideally by placing them such that internal Transporter-to-
Transporter communication routes correctly, rather than relying on
mismatched internal/external IP paths across network boundaries.
""",
    },
    {
        "id": "nakivo-backup-job-fails-older-cpu-sse42",
        "title": "NAKIVO: Backup Job Fails on Older CPUs Lacking SSE4.2 Support",
        "url": "https://helpcenter.nakivo.com/Knowledge-Base/Content/Director-and-Transporter-Issues/VMware-Backup-Job-Fails-on-Older-CPUs.htm",
        "category": "Backup & Disaster Recovery",
        "content": """# NAKIVO: Backup Job Fails on Older CPUs Lacking SSE4.2 Support

## Problem
On hosts running older CPUs without SSE4.2 instruction set support, the
Onboard Transporter's core backup process crashes during job processing
when using FAST compression (which relies on Snappy + SSE4.2). This makes
the Transporter service become unavailable and the backup job fails with a
connection error.

## Affected Scope
This can affect any job type relying on the Onboard Transporter, including
regular backup jobs and backup copy jobs.

## Remediation Options
1. **Preferred fix:** Upgrade NAKIVO Backup & Replication to a version that
   resolves the SSE4.2 dependency issue.
2. **Workaround if upgrading isn't immediately possible:** Change the
   backup repository's compression level from FAST to MEDIUM or BEST —
   these modes don't rely on SSE4.2 and avoid the crash.
3. **Alternative workaround:** Run the job using a separate, non-onboard
   Transporter installed on a different Windows machine instead of the
   Onboard Transporter on the affected host.
""",
    },
    {
        "id": "nakivo-cifs-repository-connection-lost",
        "title": "NAKIVO: Backup Job Fails When CIFS Repository Connection Is Lost",
        "url": "https://helpcenter.nakivo.com/Knowledge-Base/Content/Supported-Platforms-Issues/Backup-Job-Fails-When-CIFS-Repository-Connection-Is-Lost.htm",
        "category": "Backup & Disaster Recovery",
        "content": """# NAKIVO: Backup Job Fails When CIFS Repository Connection Is Lost

## Problem
A backup job targeting a CIFS (network share) repository behaves
inconsistently when the connection to that repository drops mid-job,
depending on which OS the Transporter runs on.

## Behavior Differences by Transporter OS
- **Windows Transporter:** The job fails outright with a clear error when
  the CIFS repository becomes unreachable — this is the expected, easier
  to monitor behavior.
- **Linux Transporter:** The job can remain in a "running" state
  indefinitely instead of failing, resuming automatically if connectivity
  is restored, or continuing to appear stuck if it isn't. This
  inconsistency can make outages harder to detect via job status alone.

## Recommendation
Where possible, use a Windows-based Transporter for CIFS repositories to
get consistent, predictable failure behavior. If a Linux Transporter must
be used, be aware that a "running" job status doesn't guarantee the job is
actually making progress — verify actual repository connectivity directly
rather than relying solely on job state.
""",
    },
    {
        "id": "nakivo-transporter-read-write-issues",
        "title": "NAKIVO: Troubleshooting Transporter Read/Write Issues to Backup Repository",
        "url": "https://helpcenter.nakivo.com/Knowledge-Base/Content/Director-and-Transporter-Issues/Transporter-Read-Write-Issues.htm",
        "category": "Backup & Disaster Recovery",
        "content": """# NAKIVO: Troubleshooting Transporter Read/Write Issues

## Problem
A Transporter is unable to properly read from or write to a Backup
Repository during a job, resulting in job failure or stalled progress.

## Diagnostic and Remediation Steps

### Step 1: Confirm Repository Accessibility
Verify the Transporter machine can actually reach and access the Backup
Repository location (network share, local disk, or cloud target).

### Step 2: Check Permissions
Confirm the account or service running the Transporter has the necessary
read/write permissions on the repository location.

### Step 3: Restart the Transporter Service
A service restart can clear transient file-handle or connection issues.

### Step 4: Verify Free Space
Ensure the Backup Repository has sufficient free space — a repository at or
near capacity can manifest as a read/write failure rather than a clear
"disk full" error.
""",
    },
    {
        "id": "nakivo-corrupted-backup-object-disk",
        "title": "NAKIVO: Corrupted Backup Object Disk Issue on Cloud Repositories",
        "url": "https://helpcenter.nakivo.com/Knowledge-Base/Content/Backup-Repository-Issues/Corrupted-Backup-Object-Disk-Issue.htm",
        "category": "Backup & Disaster Recovery",
        "content": """# NAKIVO: Corrupted Backup Object Disk Issue on Cloud Repositories

## Problem
The Transporter fails to properly write data to the Backup Repository, or
fails to move data from a transit folder into its final backup object
location — most commonly observed with cloud-based repositories such as
Amazon S3 or Wasabi.

## Likely Causes
- Intermittent connectivity issues to the cloud storage endpoint during a
  write or move operation.
- Permission errors on the cloud storage bucket/target.
- Storage limitations or quota issues on the cloud target.
- An invalid or expired SSL/TLS certificate on the cloud repository
  endpoint.
- DNS resolution problems on the Transporter host.

## Remediation Steps
1. Confirm the SSL/TLS certificate used by the cloud repository (S3,
   Wasabi, etc.) is valid and not expired.
2. Verify the Transporter host has correct, working DNS configuration.
3. In the repository's settings, under its Options tab, enable "Enforce
   explicit file system sync" to reduce the chance of incomplete writes.
4. Reduce concurrency: edit the affected backup job's Options and set the
   Transporter load to 1 concurrent task at a time, to reduce potential
   file system conflicts during the write/move operation.
""",
    },
    {
        "id": "nakivo-external-database-connection-issues",
        "title": "NAKIVO: Troubleshooting External Database Connection Issues",
        "url": "https://helpcenter.nakivo.com/display/KB/Troubleshooting+External+Database+Connection+Issues",
        "category": "Backup & Disaster Recovery",
        "content": """# NAKIVO: Troubleshooting External Database Connection Issues

## Problem
NAKIVO Backup & Replication, configured to use an external PostgreSQL
database, loses connection to that database or is otherwise unable to
communicate with it properly — which can also manifest as an inability to
create or edit tenants in Multi-Tenant mode.

## Remediation Steps

### Step 1: Update Database Connection Info
Stop NAKIVO Backup & Replication, locate the configuration file at
`[Product location]/userdata/config.properties`, and update the database
host, port, database name, username, and password entries in the
`database.url` and related keys as needed.

### Step 2: Restart Services
Restart the external database (on the same or a different host, as
applicable), then restart NAKIVO Backup & Replication.

### Step 3: Check Connection Limits
If the issue involves an exceeded number of concurrent connections,
increase the database's `max_connections` setting to match or exceed the
number of active NAKIVO jobs, then restart the database host.

## If the Database Was Broken
If the external database itself was corrupted beyond repair, install a new
instance of NAKIVO Backup & Replication, point it to a new external
database, and restore configuration from a self-backup or export bundle
rather than attempting to repair the broken database in place.
""",
    },
]


def build_docs():
    out_dir = "data/raw/nakivo"
    os.makedirs(out_dir, exist_ok=True)
    for item in NAKIVO_DOCS:
        doc = {
            "id": item["id"],
            "title": item["title"],
            "content": item["content"].strip(),
            "source": "NAKIVO Help Center",
            "url": item["url"],
            "metadata": {"category": item["category"], "platform": "nakivo/backup"},
        }
        path = os.path.join(out_dir, f"{item['id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        print(f"Wrote {path}")


if __name__ == "__main__":
    build_docs()