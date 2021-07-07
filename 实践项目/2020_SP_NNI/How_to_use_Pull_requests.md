# 如何使用 Pull Request 进行团队协作

Github 提供了 Pull Request 功能，通过该功能，团队成员之间可以方便地进行协作，进行 Code Review 来保证代码的稳定性和健壮性。以 ai-edu 为例，本文介绍如何使用 Pull Request 进行团队协作。

术语：

- 主仓库：指 ai-edu 组织下维护的远程仓库
- 自己 Fork 的仓库：指自己通过 Github 的 Fork 功能 Fork 到自己目录下的仓库
- 本地仓库：指通过 `git clone` 从自己的 Fork 的仓库 clone 到本地的仓库

具体流程如下：

## 将主仓库 Fork 到自己的目录下

进入[主仓库](https://github.com/microsoft/ai-edu)，点击页面右上角的 Fork 按钮即可将主仓库 Fork 到自己的账号下，此时你便拥有了一份跟主仓库当前状态完全同步的仓库。

## 将自己 Fork 的仓库 Clone 到本地

使用 Git Bash 或其他类似软件将自己账号下的远程仓库 clone 到本地。以用户名为 example 的账号为例：

```bash
git clone https://github.com/example/ai-edu.git
```

之后在进行开发时，我们的工作流程为：

1. 本地新建针对某一功能的分支，并编写内容
2. 将该分支的修改 commit 到本地仓库
3. 将本地完成编码的分支 push 到自己 fork 的远程仓库（2、3步可重复多次）
4. 当完成功能开发的时候，需要先将自己远程仓库的分支与主仓库的某一分支同步，若有冲突则要解决冲突，之后可以向主仓库的特定分支提起 Pull Request
5. 团队成员负责 review 相关代码，若没有问题，则可以 approve 该修改，特定数目的成员批准后即可将该 Pull Request 合并到主仓库。若发现问题（如 BUG，代码风格等），则可以在对应的代码进行评论，并与作者进行交流。PR 发起者可以根据意见在本地仓库进行修改，并再次 push 到自己 Fork 的远程仓库的对应分支，github 检测到这一修改后会自动同步 PR 内的修改，刷新页面即可看到更新，不需要重新提交 PR 。
6. 项目管理者应当负责在 PR 通过 Code Review 之后将其 merge 到主仓库中
7. 删除无用分支


## 完整的 Pull Request 过程

下面以将本文档添加到主仓库为例，展示一个完整的 Pull Request 过程。

### 1. 新建本地分支

在开发之前，我们通常希望自己的本地仓库的 master 分支尽量与主仓库的 master 分支保持同步（同步流程可参考[这里](#本地仓库/自己的远程仓库与主仓库的同步)），因此，我们不应直接在 master 分支上进行开发。如果准备开发某个功能，请先新建一个分支并在该分支进行开发。

新建一个分支并切换到该分支

```bash
git checkout -b doc-how-to-use-pr
```

注意，该命令需要在项目的本地目录中使用。分支名的选取尽量保留一定的含义，例如，如果该分支是添加了一个功能，可以用 `dev-xxx` 作为分支名，如果是文档更新，可以用 `doc-xxx` 作为分支名。

### 2. 编写内容并 commit 到本地仓库

在这个分支中，我想要增加的内容就是此文档。我可以在 vscode 或其他 IDE 中打开该项目，并添加这个文件，进行相关编辑。
在编辑完成并保存后，如果键入下面的命令

```bash
git status
```

会发现如下提示

```bash
On branch doc-how-to-use-pr
Untracked files:
  (use "git add <file>..." to include in what will be committed)
        How_To_Cooperate_with_Pull_Request.md

nothing added to commit but untracked files present (use "git add" to track)
```

此时文件名是红色的，代表这个文件虽然被修改了，但是还没有加入到暂存区，无法被 git 追踪
因此，我们需要键入下面的命令

```bash
git add .
```

这条命令代表将当前目录所有未追踪的文件加入暂存区，你当然也可以指定某个文件，例如：

```bash
git add How_to_use_Pull_requests.md
```

再次执行 git status，会发现文件名已经变成了绿色，提示这是一个 new file，下一步应该将这个修改进行 commit

```bash
On branch doc-how-to-use-pr
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   How_To_Cooperate_with_Pull_Request.md
```

因此，下一步我们将该文件 commit 到本地仓库

```bash
git commit -m 'add doc about how to create PR'
```

其中 `-m` 参数后面接对这个 Pull Request 的注释，这部分应尽量表现出这个 commit 所完成的功能。如果你是第一次在本地进行 commit，可能需要设置 git 的用户和 email，根据提示进行设置并再次 commit 即可。

### 3. push 到自己 fork 的远程仓库

下一步，将该分支 push 到你 Fork 的仓库，如果你在本地新建了分支，而远端仓库此时没有这个分支，则可能提交失败。例如使用

```bash
git push
```

会提示

```bash
fatal: The current branch doc-how-to-use-pr has no upstream branch.
To push the current branch and set the remote as upstream, use

    git push --set-upstream origin doc-how-to-use-pr
```

根据提示修改相应命令

```bash
git push --set-upstream origin doc-how-to-use-pr
```

该命令会在自己 Fork 的远程仓库新建一个和本地分支同名的分支，并将本地分支 push 到该分支。注意，在进行 push 时，尽量指定对应的主机名和分支名，如果之后还要对该分支进行修改，使用

```bash
git push origin doc-how-to-use-pr
```

而不是

```bash
git push
```

其中 origin 代表主机名（即远程仓库），`doc-how-to-use-pr` 是远程仓库的分支名。尽管二者的效果相同，但 push 时尽量指定主机名和分支名以避免可能的问题。

### 4. 与主仓库同步后发起 Pull Request

完成某一个功能开发的时候，需要先将自己远程仓库的分支与主仓库的某一分支同步（具体流程可见[这里](#本地仓库/自己的远程仓库与主仓库的同步)），若有冲突则要解决冲突。

同步后即可向主仓库发起 Pull Request：首先进入自己 Fork 的仓库，左上角的分支选择自己刚刚通过 push 进行修改过的分支，然后点击右方的 Pull Request 按键即可向主仓库发起 Pull Request。注意要指定自己仓库的分支以及主仓库的对应分支。Pull Request 的标题和描述应当反映该 PR 所做的工作或者作出的修改，尽量提供更多信息来帮助 reviewer 作出判断。

### 5. Code Review

在 PR 提出后，应当指定 reviewer 对这部分修改进行代码审查，以保证最终 merge 到主仓库的代码没有问题。reviewer 在代码审查完成之后若没有问题即可批注这些修改，反之应当要求作者修改代码。在指定数量的 reviewer 均批准修改后即可将这个 PR 合并到主仓库中。若作者在提交 PR 之后发现代码中有问题，也可将 PR 的状态改为 Draft，在修改完成后重新设置为 Ready for review 的状态。

### 6. Merge

项目管理者应当负责在 PR 通过 Code Review 之后将其 merge 到主仓库中。

### 7. 删除无用分支

当实现了某个功能并 merge 到主仓库之后，我们为这个功能新建的分支便没有用处了，可以将其删除。

下次开发时重复以上的流程。

## 本地仓库/自己的远程仓库与主仓库的同步

由于多人协作开发，在其他人对主仓库进行修改后可能造成自己本地仓库以及账号下 Fork 的仓库在进度上落后于主仓库。因此，需要定期同步主仓库。

首先为本地仓库设置 upstream

```bash
git remote add upstream https://github.com/microsoft/ai-edu.git
```

这条命令为本地仓库增加了除 origin 之外追踪的另一个名为 upstream 的远程仓库，指向的是主仓库。

之后拉取主仓库的代码合并到本地仓库。以本地仓库的 master 与主仓库的 master 分支同步为例，执行以下命令：

```bash
git checkout master # 本地分支切换到master
git fetch upstream master # 拉取主仓库的 master 分支的代码
git merge upstream/master # 将拉取下来的主仓库的 master 分支合并到本地仓库的分支
git push origin master # 将合并后的 master 分支 push 到自己 Fork 的仓库以实现同步
```

## 参考资料

更多参考资料可以参见[这里](https://git-scm.com/book/zh/v2/)
