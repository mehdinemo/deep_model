SELECT
	UserID,
	Word,
	SUM ( MessageCount ) AS MessageCount
FROM
	[Concepts].[dbo].[AllKeywords]
WHERE
	UserID IN ( SELECT FK_UserId FROM [Classification].[dbo].[TwitterUserTags] WHERE FK_TagId = 10120000 )
GROUP BY
	UserID,
	Word