
Workgroup vs. Team

短板效应



乐队模式





Inner and Outer Loop
For every service, library, or other feature involving writing code, the software development process for the Engineering Knowledge team should be set up as follows:

Check-out source code from a remote repository to a local repository.
Create a new branch for your changes.
Develop (inner-loop)
Write, run, and debug your code and configuration changes.
Commit your code and push it to the remote repository.
Create a pull-request to merge your branch into a target branch.
An automated build and test pipeline validates your branch.
Optionally, an automated deployment pipeline releases the PR branch to a testing environment.
The pull-request is reviewed by multiple reviewers.
Complete your pull-request.
An automated build and test pipeline validates the updated main branch.
An automated deployment pipeline releases the updated main branch to production.

内部和外部回路

对于涉及编写代码的每个服务、库或其他功能，工程知识团队的软件开发过程应按如下方式设置：



将源代码从远程存储库签出到本地存储库。

为您的更改创建一个新分支。

开发（内部循环）

编写、运行和调试您的代码和配置更改。

提交代码并将其推送到远程存储库。

创建一个pull请求以将分支合并到目标分支中。

自动构建和测试管道将验证您的分支。

可选地，自动化部署管道将PR分支发布到测试环境。

拉取请求由多个审阅者审阅。

完成拉取请求。

自动构建和测试管道验证更新后的主分支。

自动化部署管道将更新后的主分支发布到生产。

