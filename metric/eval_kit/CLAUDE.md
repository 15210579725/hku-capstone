# eval_kit 项目进度

## 2026-09-13 — 删除仓库中的 API key

- 目标：清除 Qwen embedding 和 LLM judge 的预填 API key，并推送 GitHub。
- 修改文件：`config.json` 的 `embedding.api_key`、`judge.api_key` 均置为空字符串，更新配置说明；`README.md` 删除预填 key / 共享额度说明，改为仓库外配置副本配合 `--config` 使用；新增本进度文件。
- 关键决策：保留模型与端点。仅提交凭据清理及对应文档，本地原有并发、batch_size、评测代码和其他项目改动不纳入此提交。该阶段未重写 Git 历史；用户随后授权的历史清理见下文。
- 验证：工作区和暂存配置均可解析且两个 key 为空；当前提交全仓按原 key 精确检索仅命中本配置，清理后对待提交树复扫；eval_kit 密钥模式扫描和 Python AST 检查通过。系统 Python 缺少 numpy，改用已有 `caption/.venv-mimo/bin/python`，8 条示例的 `--dry-run` 通过，未请求外部 API。
- 未完成项：旧 Git 历史仍含曾提交的凭据；服务商后台作废/轮换未执行。
- 下一步：提交并推送，核验远端 main 的 SHA 及配置空值。以后不要提交真实 API key。

## 2026-09-13 — Git 历史中的两把 API key 清理及交付

- 目标：按用户追加要求，从历史中移除上述两把 key，并重写 GitHub `main`。
- 清理：在隔离 Git 副本中使用 git-filter-repo 2.47.0，按完整密钥精确替换文件内容和提交消息。扫描全部 83,432 个可达对象（13,119,134,112 bytes），两把 key 的残留匹配为 0；26 个发生内容变化的历史文件版本均核验为仅删除原密钥，清理前后 main 最新文件树相同。
- 交付：GitHub `main` 已从旧历史切换到 `89846c6128d5a17dc3eaf886b985768ba5d8c3f8`；通过 API 核验远端 SHA 和配置中的两处空 key。GitHub 当时只有 main、0 forks、0 PR；用于上传的临时清理分支已删除。
- 推送问题：普通 Git 历史重写推送打包 44,799 个对象（约 1.8 GB 未压缩内容）并触发 HTTP 408；实际新增对象仅 35 个、15,605 bytes。改用 GitHub Git database API 上传这些对象，逐一核验精确 SHA，再借助临时干净分支进行带旧 SHA 校验的 force-with-lease 推送；没有发布本地 stash 或内部引用。
- 本地同步：main、origin/main 及 stash 引用切换到已清理对象；仅导入 58 个对象、43,680 bytes 的 pack。同步前后原有 88 个已修改文件的 SHA-256 一致，暂存区未变；所有本地可达对象与已全量扫描的清理副本完全一致。工作区原有评测并发、batch_size、caption 等改动保留。
- 修改文件：本次交付补记仅更新本文件；历史内容清理已单独完成。用户正在自行处理 RightCode 密钥轮换，尚未收到作废确认。
- 保密边界：历史重写不能撤回第三方已下载的副本；服务商后台作废状态、GitHub 缓存清理均不能由此次推送证明。原本地仓库的不可达对象和 reflog 也未做物理清除。
- 下一步：确认两把旧 key 均已在服务商后台作废；协作者重新克隆，避免把旧历史合并推回。需要清理 GitHub 旧对象/缓存时由仓库所有者联系 GitHub Support。
- 额外验证：重写后仍能通过 GitHub API 按旧提交 SHA 读取旧提交对象；因此不能声称旧 GitHub 对象已被彻底删除。
