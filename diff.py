import filecmp

trainset_dircmp = filecmp.dircmp('data/out/trainset/gt', 'data/out/trainset/source')
testset_dircmp = filecmp.dircmp('data/out/testset/gt', 'data/out/testset/source')

stack = [trainset_dircmp, testset_dircmp]

while len(stack) != 0:
    cmp = stack.pop()
    print(cmp.left, cmp.right)
    print(cmp.left_only, cmp.right_only)
    print()
    for dir in cmp.subdirs:
        stack.insert(0, cmp.subdirs[dir])
