# eval_kit 项目进度

## 2026-09-13 — 删除仓库中的 API key

- 目标：清除 Qwen embedding 和 LLM judge 的预填 API key，并推送 GitHub。
- 修改文件：`config.json` 的 `embedding.api_key`、`judge.api_key` 均置为空字符串，更新配置说明；`README.md` 删除预填 key / 共享额度说明，改为仓库外配置副本配合 `--config` 使用；新增本进度文件。
- 关键决策：保留模型与端点。仅提交凭据清理及对应文档，本地原有并发、batch_size、评测代码和其他项目改动不纳入此提交。不重写 Git 历史。
- 验证：工作区和暂存配置均可解析且两个 key 为空；当前提交全仓按原 key 精确检索仅命中本配置，清理后对待提交树复扫；eval_kit 密钥模式扫描和 Python AST 检查通过。系统 Python 缺少 numpy，改用已有 `caption/.venv-mimo/bin/python`，8 条示例的 `--dry-run` 通过，未请求外部 API。
- 未完成项：旧 Git 历史仍含曾提交的凭据；服务商后台作废/轮换未执行。
- 下一步：提交并推送，核验远端 main 的 SHA 及配置空值。以后不要提交真实 API key。
