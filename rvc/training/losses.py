import torch


def feature_loss(fmap_r, fmap_g):
    """Compute the feature loss between reference and generated feature maps.

    Returns:
        (total_loss, fm_per_disc): total + per-sub-discriminator list
    """
    loss = 0
    fm_per_disc = []
    for dr, dg in zip(fmap_r, fmap_g):
        disc_loss = 2 * sum(torch.mean(torch.abs(rl - gl)) for rl, gl in zip(dr, dg))
        loss += disc_loss
        fm_per_disc.append(disc_loss)
    return loss, fm_per_disc


def discriminator_loss(
    disc_real_outputs, disc_generated_outputs, d_label_smoothing=0.0
):
    """Compute the discriminator loss for real and generated outputs with optional label smoothing.

    Returns:
        (total_loss, r_losses, g_losses): total loss + per-sub-discriminator lists
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1.0 - d_label_smoothing - dr.float()) ** 2)
        g_loss = torch.mean((0.0 + d_label_smoothing - dg.float()) ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss)
        g_losses.append(g_loss)
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """Compute the generator loss based on discriminator outputs.

    Args:
        disc_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.

    """
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1 - dg.float()) ** 2)
        loss += l
    return loss


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """Compute the Kullback-Leibler divergence loss.

    Args:
        z_p (torch.Tensor): Latent variable z_p [b, h, t_t].
        logs_q (torch.Tensor): Log variance of q [b, h, t_t].
        m_p (torch.Tensor): Mean of p [b, h, t_t].
        logs_p (torch.Tensor): Log variance of p [b, h, t_t].
        z_mask (torch.Tensor): Mask for the latent variables [b, h, t_t].

    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    kl = (kl * z_mask).sum()
    loss = kl / z_mask.sum()
    return loss
