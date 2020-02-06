SELECT
	DISTINCT (a.FK_UserId)
FROM
	( SELECT FK_UserId FROM [dbo].[TwitterUserDegrees] WHERE RetweetOutDegree >= 3 AND LikeOutDegree >= 5 ) a
	INNER JOIN (
SELECT
	FK_UserId,
	FK_TagId 
FROM
	[dbo].[TwitterUserMemberships] 
WHERE
	( FK_TagId = 7000000 AND Membership > 0.2 ) 
	OR ( FK_TagId = 8000000 AND Membership > 0.3 )) b ON a.FK_UserId= b.FK_UserId