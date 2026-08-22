import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/habitat-lab-v0.3.1-multi-agent.patch"
SETUP = ROOT / "scripts/setup_lite_multiagent_habitat.sh"
RUNNER = ROOT / "scripts/run-lite-multiagent.sh"
HABITAT_BASE = "142616776544f918c19e7f0392b65cc8cc69fa13"
PATCHED_PATHS = (
    "habitat-lab/habitat/config/default_structured_configs.py",
    "habitat-lab/habitat/core/simulator.py",
    "habitat-lab/habitat/sims/habitat_simulator/__init__.py",
    "habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py",
    "habitat-lab/habitat/sims/habitat_simulator/multi_agent_sim.py",
)


class LiteMultiAgentRuntimeSetupTests(unittest.TestCase):
    def run_command(self, *args, **kwargs):
        return subprocess.run(
            args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            **kwargs,
        )

    def test_patch_is_pinned_and_contains_complete_multiagent_simulator_change(self):
        """Catches a non-reproducible patch that omits an edited or new Habitat file."""
        content = PATCH.read_text(encoding="utf-8")
        self.assertIn(f"base commit {HABITAT_BASE}", content)
        for relative_path in PATCHED_PATHS:
            self.assertIn(f"a/{relative_path}", content)
            self.assertIn(f"b/{relative_path}", content)
        self.assertIn("@registry.register_simulator(name=\"MultiAgentSim-v0\")", content)

    def test_setup_applies_patch_to_clean_pinned_worktree_and_is_idempotent(self):
        """Catches setup mutating the live checkout or failing on a second invocation."""
        source_checkout = ROOT / "habitat-lab"
        if not source_checkout.is_dir():
            self.skipTest("the ignored Habitat-Lab checkout is unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            worktree = Path(temp_dir) / "habitat-lab"
            created = self.run_command(
                "git",
                "-C",
                str(source_checkout),
                "worktree",
                "add",
                "--detach",
                str(worktree),
                HABITAT_BASE,
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            try:
                env = os.environ | {
                    "APEXNAV_ROOT": str(ROOT),
                    "HABITAT_LAB_DIR": str(worktree),
                    "PATCH_FILE": str(PATCH),
                }
                first = self.run_command("bash", str(SETUP), env=env)
                self.assertEqual(first.returncode, 0, first.stderr)
                self.assertIn("Applied", first.stdout)
                for relative_path in PATCHED_PATHS:
                    self.assertTrue((worktree / relative_path).is_file(), relative_path)
                reverse_check = self.run_command(
                    "git", "-C", str(worktree), "apply", "--unidiff-zero", "--reverse", "--check", str(PATCH)
                )
                self.assertEqual(reverse_check.returncode, 0, reverse_check.stderr)
                second = self.run_command("bash", str(SETUP), env=env)
                self.assertEqual(second.returncode, 0, second.stderr)
                self.assertIn("already applied", second.stdout)
            finally:
                removed = self.run_command(
                    "git", "-C", str(source_checkout), "worktree", "remove", "--force", str(worktree)
                )
                self.assertEqual(removed.returncode, 0, removed.stderr)

    def test_setup_rejects_a_checkout_at_the_wrong_base_with_recovery_guidance(self):
        """Catches accidental application of a v0.3.1 patch to an unknown Habitat revision."""
        source_checkout = ROOT / "habitat-lab"
        if not source_checkout.is_dir():
            self.skipTest("the ignored Habitat-Lab checkout is unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            worktree = Path(temp_dir) / "habitat-lab"
            created = self.run_command(
                "git", "-C", str(source_checkout), "worktree", "add", "--detach", str(worktree), HABITAT_BASE
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            try:
                configured = self.run_command("git", "-C", str(worktree), "config", "user.email", "test@example.invalid")
                self.assertEqual(configured.returncode, 0, configured.stderr)
                configured = self.run_command("git", "-C", str(worktree), "config", "user.name", "Runtime test")
                self.assertEqual(configured.returncode, 0, configured.stderr)
                advanced = self.run_command("git", "-C", str(worktree), "commit", "--allow-empty", "-m", "wrong base")
                self.assertEqual(advanced.returncode, 0, advanced.stderr)
                env = os.environ | {
                    "APEXNAV_ROOT": str(ROOT),
                    "HABITAT_LAB_DIR": str(worktree),
                    "PATCH_FILE": str(PATCH),
                }
                result = self.run_command("bash", str(SETUP), env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(HABITAT_BASE, result.stderr)
                self.assertIn("git -C", result.stderr)
                self.assertIn("checkout --detach", result.stderr)
            finally:
                removed = self.run_command(
                    "git", "-C", str(source_checkout), "worktree", "remove", "--force", str(worktree)
                )
                self.assertEqual(removed.returncode, 0, removed.stderr)

    def test_setup_rejects_staged_tracked_changes_before_applying_patch(self):
        """Catches setup applying the patch over an unrelated staged user edit."""
        source_checkout = ROOT / "habitat-lab"
        if not source_checkout.is_dir():
            self.skipTest("the ignored Habitat-Lab checkout is unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            worktree = Path(temp_dir) / "habitat-lab"
            created = self.run_command(
                "git", "-C", str(source_checkout), "worktree", "add", "--detach", str(worktree), HABITAT_BASE
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            try:
                readme = worktree / "README.md"
                readme.write_text(readme.read_text(encoding="utf-8") + "\nStaged test edit.\n", encoding="utf-8")
                staged = self.run_command("git", "-C", str(worktree), "add", "README.md")
                self.assertEqual(staged.returncode, 0, staged.stderr)
                env = os.environ | {
                    "APEXNAV_ROOT": str(ROOT),
                    "HABITAT_LAB_DIR": str(worktree),
                    "PATCH_FILE": str(PATCH),
                }
                result = self.run_command("bash", str(SETUP), env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("local changes", result.stderr)
                self.assertIn("status --short", result.stderr)
                self.assertFalse((worktree / PATCHED_PATHS[-1]).exists())
            finally:
                removed = self.run_command(
                    "git", "-C", str(source_checkout), "worktree", "remove", "--force", str(worktree)
                )
                self.assertEqual(removed.returncode, 0, removed.stderr)

    def test_runner_uses_lite_environment_caches_ros_and_local_habitat_first(self):
        """Catches the packaged runner selecting an installed Habitat or downloading Lite models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            project_root = temp / "ApexNav"
            lite_root = temp / "Lite-ApexNav"
            python_path = lite_root / "conda-env/Lite-apex/bin/python"
            habitat_path = project_root / "habitat-lab/habitat-lab"
            ros_setup = temp / "ros/setup.bash"
            devel_setup = project_root / "devel/setup.bash"
            python_path.parent.mkdir(parents=True)
            habitat_path.mkdir(parents=True)
            ros_setup.parent.mkdir(parents=True)
            devel_setup.parent.mkdir(parents=True)
            ros_setup.write_text("export ROS_SETUP_MARKER=loaded\n", encoding="utf-8")
            devel_setup.write_text("export DEVEL_SETUP_MARKER=loaded\n", encoding="utf-8")
            python_path.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'pythonpath=%s\\n' \"$PYTHONPATH\"\n"
                "printf 'yoloe=%s\\n' \"$YOLOE_WEIGHTS\"\n"
                "printf 'clip=%s\\n' \"$CLIP_DOWNLOAD_ROOT\"\n"
                "printf 'attempts=%s timeout=%s backoff=%s\\n' \"$VLM_REQUEST_ATTEMPTS\" \"$VLM_REQUEST_TIMEOUT\" \"$VLM_RETRY_BACKOFF\"\n"
                "printf 'setup=%s/%s\\n' \"$ROS_SETUP_MARKER\" \"$DEVEL_SETUP_MARKER\"\n"
                "printf 'args=%s\\n' \"$*\"\n",
                encoding="utf-8",
            )
            python_path.chmod(0o755)
            env = os.environ | {
                "APEXNAV_ROOT": str(project_root),
                "LITE_APEX_ROOT": str(lite_root),
                "ROS_SETUP": str(ros_setup),
            }
            result = self.run_command("bash", str(RUNNER), "arbitrary.py", "--flag", "value", env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"pythonpath={habitat_path}:", result.stdout)
            self.assertIn(f"yoloe={lite_root}/model-cache/yoloe-11l-seg.pt", result.stdout)
            self.assertIn(f"clip={lite_root}/model-cache/clip", result.stdout)
            self.assertIn("attempts=1 timeout=5 backoff=0.5", result.stdout)
            self.assertIn("setup=loaded/loaded", result.stdout)
            self.assertIn("args=arbitrary.py --flag value", result.stdout)

    def test_operating_guide_uses_only_lite_services_and_explains_current_frame_semantics(self):
        """Catches documentation reviving retired services or promising a nonexistent VLM queue."""
        guide = (ROOT / "liteOp.txt").read_text(encoding="utf-8")
        self.assertIn("12182", guide)
        self.assertIn("12184", guide)
        self.assertIn("YOLOE", guide)
        self.assertIn("CLIPITM", guide)
        self.assertIn("hm3dv1", guide)
        self.assertIn("hm3dv2", guide)
        self.assertIn("mp3d", guide)
        self.assertIn("/healthz", guide)
        self.assertIn("当前帧", guide)
        self.assertIn("内部队列", guide)
        self.assertIn("Habitat client", guide)
        self.assertIn("Flask server", guide)
        self.assertIn("request_lock", guide)
        self.assertIn("外部并发请求可能等待", guide)
        self.assertNotIn("后台积压", guide)
        self.assertNotIn("grounding_dino", guide)
        self.assertNotIn("blip2itm", guide)


if __name__ == "__main__":
    unittest.main()
